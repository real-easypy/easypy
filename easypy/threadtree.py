"""
This module monkey-patches python's (3.x) threading module so that we can find which thread spawned which.
This is useful in our contextualized logging, so that threads inherit logging context from their 'parent'.
"""
import sys
from collections import defaultdict
from importlib import import_module
from weakref import WeakKeyDictionary, WeakSet
import time
import logging
from contextlib import contextmanager
import _thread
import threading

from easypy.gevent import main_thread_ident_before_patching, is_module_patched
from easypy.bunch import Bunch
from easypy.collections import ilistify
from easypy._multithreading_init import UUIDS_TREE, IDENT_TO_UUID, UUID_TO_IDENT, MAIN_UUID, _BOOTSTRAPPERS, get_thread_uuid


_REGISTER_GREENLETS = False
_FRAME_SNAPSHOTS_REGISTRY = defaultdict(dict)

_orig_start_new_thread = _thread.start_new_thread


def start_new_thread(target, *args, **kwargs):
    """
    A wrapper for the built-in 'start_new_thread' used to capture the parent of each new thread.
    """
    parent_thread = threading.current_thread()
    parent_uuid = get_thread_uuid(parent_thread)
    parent_frame = sys._getframe(0)

    tc_fork = ThreadContexts.fork(parent_uuid)
    tc_fork.send(None)

    def wrapper(*args, **kwargs):
        nonlocal parent_thread, parent_uuid, parent_frame
        thread = threading.current_thread()
        uuid = get_thread_uuid(thread)
        tc_fork.send(uuid)
        _FRAME_SNAPSHOTS_REGISTRY[parent_thread.ident][thread.ident] = parent_frame

        UUIDS_TREE[uuid] = parent_uuid
        if _REGISTER_GREENLETS:
            IDENT_TO_GREENLET[thread.ident] = gevent.getcurrent()
        try:
            return target(*args, **kwargs)
        finally:
            IDENT_TO_UUID.pop(thread.ident)

    return _orig_start_new_thread(wrapper, *args, **kwargs)


get_parent_uuid = UUIDS_TREE.get


def get_thread_parent(thread):
    """
    Returns parent thread for the given thread.
    If the parent thread died, returns a ``DeadThread`` object, to preserve the thread-tree structure.
    """
    uuid = thread.uuid
    parent_uuid = UUIDS_TREE.get(uuid)
    if not parent_uuid:
        return

    # double check that values synchronized while locked
    with threading._active_limbo_lock:
        parent_ident = UUID_TO_IDENT.get(parent_uuid)
        uuid_at_ident = IDENT_TO_UUID.get(parent_ident)
        if uuid_at_ident != parent_uuid:
            # a new thread assumed this ident[ity], which is not the original parent
            return DeadThread.get(parent_uuid)

        return threading._active.get(parent_ident) or DeadThread.get(parent_uuid)


_thread.start_new_thread = start_new_thread
threading._start_new_thread = start_new_thread


class DeadThread(object):
    """
    A dummy object representing a thread that already died.
    """

    name = "DeadThread"
    ident = 0
    uuid = 0

    def __init__(self, uuid):
        self.ident = int(uuid)
        self.uuid = uuid

    def __repr__(self):
        return "DeadThread(%s)" % self.uuid

    def __eq__(self, other):
        if isinstance(other, DeadThread):
            return self.uuid == other.uuid
        return False

    def __hash__(self):
        return self.__repr__().__hash__()

    @classmethod
    def get(cls, uuid):
        return cls(uuid)


threading.Thread.parent = property(get_thread_parent)
threading.Thread.uuid = property(get_thread_uuid)

if is_module_patched("threading"):
    _REGISTER_GREENLETS = True
    from gevent import Greenlet
    import gc
    import gevent.monkey
    gevent.monkey.saved['threading']['Thread'].parent = property(get_thread_parent)
    gevent.monkey.saved['threading']['Thread'].uuid = property(get_thread_uuid)
    IDENT_TO_UUID[main_thread_ident_before_patching] = get_thread_uuid()
    from weakref import WeakValueDictionary
    IDENT_TO_GREENLET = WeakValueDictionary()
    thread = threading.current_thread()
    IDENT_TO_GREENLET[thread.ident] = gevent.getcurrent()

    def get_all_greenlets():
        found = set()
        for g in gc.get_objects():
            try:
                if isinstance(g, Greenlet):
                    found.add(g)
            except ReferenceError:
                pass
        return found

    def iter_thread_frames():
        current_thread_ident = threading.current_thread().ident

        def fix(ident, frame):
            # if it is the current thread, we mustn't yield this very frame,
            # since it'll get detached from the stack once we exit this function
            return ident, (frame.f_back if ident == current_thread_ident else frame)

        main_thread_frame = None
        for ident, frame in sys._current_frames().items():
            if IDENT_TO_UUID.get(ident) == MAIN_UUID:
                main_thread_frame = frame
                # the MainThread should be shown in it's "greenlet" version
                continue
            yield fix(ident, frame)

        all_greenlets = get_all_greenlets()

        for thread in threading.enumerate():
            greenlet = IDENT_TO_GREENLET.get(thread.ident)
            all_greenlets.discard(greenlet)
            if not greenlet:
                # some inbetween state, before greenlet started or after it died?...
                pass
            elif greenlet.gr_frame:
                yield fix(thread.ident, greenlet.gr_frame)
            else:
                # a thread with greenlet but without gr_frame will be fetched from sys._current_frames
                # If we switch to another greenlet by the time we get there we will get inconsistent dup of threads.
                # TODO - make best-effort attempt to show coherent thread dump
                yield fix(thread.ident, main_thread_frame)

        for greenlet in sorted(all_greenlets, key=id):
            if greenlet.gr_frame:
                yield id(greenlet) * -1, greenlet.gr_frame

else:
    def iter_thread_frames():
        yield from sys._current_frames().items()


class FrameWrapper:
    """This wrapper adds the `f_thread_ident` attribute, which stores the thread
    identifier of the current thread. All other attribute accesses are forwarded
    to the wrapped frame object, making the wrapper behave like the original frame.
    """

    def __init__(self, frame, thread_ident):
        self._frame = frame
        self.f_thread_ident = thread_ident

    def __getattr__(self, name):
        """Delegate attribute access to the wrapped frame object."""
        return getattr(self._frame, name)

    def __setattr__(self, name, value):
        """Allow setting attributes."""
        if name == "_frame" or name == "f_thread_ident":
            super().__setattr__(name, value)  # Handle internal attributes
        else:
            setattr(self._frame, name, value)  # Delegate to the wrapped frame

    def __repr__(self):
        """Provide a string representation including the class name and f_thread_ident."""
        return (
            f"<{self.__class__.__name__}(frame={repr(self._frame)}, "
            f"f_thread_ident={self.f_thread_ident})>"
        )

    def __str__(self):
        """Provide a readable string for the wrapper."""
        return str(self._frame)

    def __eq__(self, other):
        """Equality comparison."""
        if isinstance(other, self.__class__):
            return (
                self._frame == other._frame
                and self.f_thread_ident == other.f_thread_ident
            )
        return self._frame == other

    def __ne__(self, other):
        """Inequality comparison."""
        return not self.__eq__(other)

    def __hash__(self):
        """Make the object hashable."""
        return hash(self._frame)

    def __dir__(self):
        """List attributes of the wrapper and wrapped object."""
        return dir(self._frame) + ["f_thread_ident"]


def walk_frames(thread=None, *, across_threads=False, use_snapshots=False):
    """
    Yields the stack frames of the current/specified thread.

    :param across_threads: yield frames of ancestor threads.
    :param use_snapshots: If True, yield frames from thread snapshots, ensuring consistent frame retrieval even if the
    parent thread has moved on.

    This function traverses the stack of the given thread, optionally walking up the stack of ancestor threads if
    `across_threads` is set.
    When `use_snapshots` is enabled, frames are retrieved from snapshots, mitigating issues where the parent thread
    might no longer be at the frame that spawned the current thread.
    """

    if not thread:
        thread = threading.current_thread()

    if use_snapshots:
        frame = sys._getframe(0)
        while frame:
            yield FrameWrapper(frame, thread.ident)
            frame = frame.f_back
            if not frame and across_threads and thread.parent:
                frame = _FRAME_SNAPSHOTS_REGISTRY[thread.parent.ident][thread.ident]
                thread = thread.parent
    else:
        frame = dict(iter_thread_frames()).get(thread.ident)

        while frame:

            yield frame

            frame = frame.f_back
            if not frame and across_threads and thread.parent:
                thread = thread.parent
                frame = dict(iter_thread_frames()).get(thread.ident)


this_module = import_module(__name__)
_BOOTSTRAPPERS |= {threading, this_module}


def get_thread_trees(including_this=True):
    """
    Returns raw information about currently active threads, including their stack frames, and their immediate child-threads.

    The tree structure is maintained by special monkey-patching implemented by this module.
    """

    from .logging import THREAD_LOGGING_CONTEXT
    from .bunch import Bunch
    from .humanize import format_thread_stack

    tree = {}
    dead_threads = set()
    contexts = {}
    stacks = {}

    def add_to_tree(thread):
        contexts[thread.ident] = THREAD_LOGGING_CONTEXT.flatten(thread.uuid)
        parent = get_thread_parent(thread)
        if isinstance(parent, DeadThread) and parent not in dead_threads:
            dead_threads.add(parent)
            add_to_tree(parent)
        tree.setdefault(parent, []).append(thread)

    for thread in threading.enumerate():
        add_to_tree(thread)

    current_ident = threading.current_thread().ident
    main_ident = threading.main_thread().ident

    for thread_ident, frame in iter_thread_frames():
        if not including_this and thread_ident == current_ident:
            formatted = "  <this frame>"
        else:
            # show the entire stack if it's this thread, don't skip ('after_module') anything
            show_all = thread_ident in (current_ident, main_ident)
            formatted = format_thread_stack(frame, skip_modules=[] if show_all else _BOOTSTRAPPERS) if frame else ''
        stacks[thread_ident] = formatted, time.time(), (id(frame) if frame else 0)

    def add_thread(parent_thread, parent):
        for thread in sorted(tree[parent_thread], key=lambda thread: thread.name):
            ident = thread.ident or 0
            stack, ts, frame_id = stacks.pop(ident, ("", 0, 0))
            context = contexts.get(ident, {})
            context_line = ", ".join("%s: %s" % (k, context[k]) for k in "host context".split() if context.get(k))

            this = Bunch(
                name=thread.name,
                daemon="[D]" if getattr(thread, "daemon", False) else "",
                ident=ident,
                context_line="({})".format(context_line) if context_line else "",
                stack=stack,
                timestamp=ts,
                frame_id=frame_id,
                children=[],
            )
            parent.children.append(this)
            if thread in tree:
                add_thread(thread, this)
        return parent

    root = Bunch(children=[])
    add_thread(None, root)

    for ident, (stack, ts, frame_id) in stacks.items():
        orphan = Bunch(
            name="<Greenlet>" if ident < 0 else "<Orphan>",
            daemon="[D]",
            ident=abs(ident),
            context_line="",
            stack=stack,
            timestamp=ts,
            frame_id=frame_id,
            children=[],
        )
        root.children.append(orphan)
    return root.children


def get_thread_stacks(including_this=True):
    """
    Returns an ``IndentableTextBuffer`` which renders the currently active threads.
    """

    stack_trees = get_thread_trees(including_this=including_this)

    def write_thread(branches):
        for branch in branches:
            with buff.indent('Thread{daemon}: {name} ({ident:X})', **branch):
                ts = time.strftime("%H:%M:%S", time.localtime(branch.timestamp))
                buff.write("{} Fid:{:X}  {}", ts, branch.frame_id, branch.context_line)
                for line in branch.stack.splitlines():
                    buff.write(line)
                write_thread(branch.children)

    from .humanize import IndentableTextBuffer
    buff = IndentableTextBuffer("Current Running Threads")
    write_thread(stack_trees)

    return buff


def watch_threads(interval, logger_name='threads'):
    """
    Starts a daemon thread that logs the active-threads to a ``threads`` logger, at the specified interval.
    The data is logged using the ``extra`` logging struct.
    It is recommended to configure the logger to use ``easypy.logging.YAMLFormatter``, so that the log can be
    easily parsed by other tools.
    """

    from easypy.resilience import resilient
    from easypy.concurrency import concurrent
    from easypy.logging import _get_logger

    cmdline = " ".join(sys.argv)
    logger = _get_logger(name=__name__)
    threads_logger = _get_logger(name=logger_name)

    last_threads = set()

    @contextmanager
    def no_exceptions():
        try:
            yield
        except Exception:
            pass

    @no_exceptions()
    @resilient.warning
    def dump_threads():
        nonlocal last_threads

        with logger.indented('getting thread tree', level=logging.DEBUG):
            trees = get_thread_trees(including_this=False)

        with logger.indented('logging threads to yaml', level=logging.DEBUG):
            threads_logger.debug("threads", extra=dict(cmdline=cmdline, tree=Bunch(children=trees).to_dict()))

        with logger.indented('creating current thread set', level=logging.DEBUG):
            current_threads = set()
            for thread in threading.enumerate():
                if thread.ident:
                    current_threads.add(thread.ident)

        new_threads = current_threads - last_threads
        closed_threads = last_threads - current_threads
        stringify = lambda idents: ", ".join("%X" % ident for ident in idents)

        if new_threads:
            logger.debug("WHITE<<NEW>> threads (%s): %s", len(new_threads), stringify(new_threads))
        if closed_threads:
            logger.debug("threads terminated (%s): %s", len(closed_threads), stringify(closed_threads))
        logger.debug("total threads: %s", len(current_threads))

        last_threads = current_threads

    thread = concurrent(dump_threads, threadname="ThreadWatch", loop=True, sleep=interval, real_thread_no_greenlet=True)
    thread.start()
    logger.info("threads watcher started")


class ThreadContexts():
    """
    A structure for storing arbitrary data per thread.
    Unlike ``threading.local``, data stored here will be inherited into
    child-threads (threads spawned from threads)::

        TC = ThreadContexts()

        # (thread-a)
        with TC(my_data='b'):
            spawn_thread_b()

        # (thread-b)
        assert TC.my_data == 'b'
        with TC(my_data='c'):
            spawn_thread_c()
        assert TC.my_data == 'b'

        # (thread-c)
        assert TC.my_data == 'c'

    :param counters: attributes named here get incremented each time, instead of overwritten:
        ::
            TC = ThreadContexts(counters=('i', 'j'))
            assert TC.i == TC.j == 0

            with TC(i=1):
                assert TC.i == 1
                assert TC.j == 0

                with TC(i=1, j=1):
                    assert TC.i == 2
                    assert TC.j == 1


    :param stacks: attributes named here get pushed into a list, instead of overwritten:
        ::
            TC = ThreadContexts(stacks=('i', 'j'))
            with TC(i='a'):
                assert TC.i == ['a']
                assert TC.j == []
                with TC(i='i', j='j'):
                    assert TC.i == ['a', 'i']
                    assert TC.j == ['j']

    """

    # prevent setting custom attributes
    __slots__ = "_context_data _defaults _counters _stacks".split()

    # registry of all TCs
    _ALL = set()

    def __init__(self, defaults={}, counters=None, stacks=None):
        self._context_data = WeakKeyDictionary()
        self._defaults = defaults.copy()
        self._counters = set(ilistify(counters or []))
        self._stacks = set(ilistify(stacks or []))
        self._ALL.add(self)

    def __del__(self):
        self._ALL.discard(self)

    def update_defaults(self, **kwargs):
        self._defaults.update(kwargs)

    @classmethod
    def fork(cls, parent_thread):
        data = {}
        for tc in cls._ALL:
            # we copy the list itself, and each Bunch within it
            data[tc] = [c.copy() for c in tc._get_context_data(parent_thread)]

        child_thread = yield
        for tc in cls._ALL:
            tc._context_data[child_thread] = data[tc]
        yield

    def _get_context_data(self, thread_uuid=None):
        return self._context_data.setdefault(thread_uuid or get_thread_uuid(), [])

    def get(self, k, default=None):
        """
        safely get the value of the specified key, in the current thread context
        """
        return self.flatten().get(k, default)

    def __getattr__(self, k):
        ret = self.get(k, AttributeError)
        if ret is AttributeError:
            raise AttributeError(k)
        return ret

    @contextmanager
    def __call__(self, kw=None, **kwargs):
        kw = dict(kw or {}, **kwargs)
        for v in kw.values():
            assert (v is None) or (isinstance(v, (str, int, float))), "Can't use %r as context vars" % v
        ctx = self._get_context_data()
        ctx.append(Bunch(kw))
        try:
            yield
        except Exception as exc:
            context = self.flatten()
            if context and not getattr(exc, "context", None):
                try:
                    exc.context = context
                except:  # noqa: E722
                    logging.warning("could not attach context to exception")
            raise
        finally:
            ctx.pop(-1)

    def flatten(self, thread_uuid=None):
        """
        return a flattened dict of all inherited context data for the current thread
        """
        stack = self._get_context_data(thread_uuid=thread_uuid)
        concats = {k: self._defaults.get(k, []) for k in self._stacks}
        accums = {k: self._defaults.get(k, 0) for k in self._counters}
        extra = dict(self._defaults)
        for ctx in stack:
            for k in self._counters:
                accums[k] += ctx.get(k, 0)
            for k in self._stacks:
                if k in ctx:
                    concats[k].append(ctx[k])
            extra.update(ctx)
        extra.update(concats)
        extra.update(accums)
        return extra
