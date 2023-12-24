"""
This module monkey-patches python's (3.x) threading module so that we can find which thread spawned which.
This is useful in our contextualized logging, so that threads inherit logging context from their 'parent'.
"""
import sys
from importlib import import_module
from uuid import uuid4
from weakref import WeakKeyDictionary
import time
import logging
from copy import deepcopy
from contextlib import contextmanager
import _thread
import threading

from easypy.gevent import main_thread_ident_before_patching, is_module_patched
from easypy.bunch import Bunch
from easypy.collections import ilistify
from easypy._multithreading_init import UUIDS_TREE, IDENT_TO_UUID, UUID_TO_IDENT, MAIN_UUID, _BOOTSTRAPPERS, get_thread_uuid


_REGISTER_GREENLETS = False
_orig_start_new_thread = _thread.start_new_thread


def start_new_thread(target, *args, **kwargs):
    """
    A wrapper for the built in 'start_new_thread' used to capture the parent of each new thread.
    """
    parent_uuid = get_thread_uuid()

    def wrapper(*args, **kwargs):
        thread = threading.current_thread()
        uuid = get_thread_uuid(thread)
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


def walk_frames(thread=None, *, across_threads=False):
    """
    Yields the stack frames of the current/specified thread.

    :param across_threads: yield frames of ancestor threads.
    Note that the parent thread might be off to other things, and not actually in the frame that spawned the thread at the tip.
    """

    if not thread:
        thread = threading.current_thread()

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

    def __init__(self, defaults={}, counters=None, stacks=None):
        self._context_data = WeakKeyDictionary()
        self._defaults = defaults.copy()
        self._counters = set(ilistify(counters or []))
        self._stacks = set(ilistify(stacks or []))

    def update_defaults(self, **kwargs):
        self._defaults.update(kwargs)

    def _get_context_data(self, thread_uuid=None, combined=False):
        if not thread_uuid:
            thread_uuid = get_thread_uuid()

        ctx = self._context_data.setdefault(thread_uuid, [])
        if not combined:
            return ctx

        parent_uuid = get_parent_uuid(thread_uuid)
        if parent_uuid:
            parent_ctx = self._get_context_data(parent_uuid, combined=True)
            ctx = deepcopy(parent_ctx) + ctx
        return ctx

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
                except:
                    logging.warning("could not attach context to exception")
            raise
        finally:
            ctx.pop(-1)

    def flatten(self, thread_uuid=None):
        """
        return a flattened dict of all inherited context data for the current thread
        """
        stack = self._get_context_data(thread_uuid=thread_uuid, combined=True)
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
