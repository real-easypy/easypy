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
from logging import getLogger
import _thread
import threading

from easypy.humanize import format_thread_stack

from easypy.gevent import main_thread_ident_before_patching, is_module_patched
from .bunch import Bunch
from .collections import ilistify

from . import UUIDS_TREE, IDENT_TO_UUID, UUID_TO_IDENT, MAIN_UUID
_logger = getLogger(__name__)


def get_thread_uuid(thread=None):
    if not thread:
        thread = threading.current_thread()

    ident = thread.ident
    try:
        uuid = IDENT_TO_UUID[ident]
    except KeyError:
        uuid = IDENT_TO_UUID.setdefault(ident, uuid4())
        UUID_TO_IDENT[uuid] = ident
    return uuid

_orig_start_new_thread = _thread.start_new_thread


def start_new_thread(target, *args, **kwargs):
    parent_uuid = get_thread_uuid()

    def wrapper(*args, **kwargs):
        thread = threading.current_thread()
        uuid = get_thread_uuid(thread)
        UUIDS_TREE[uuid] = parent_uuid
        try:
            return target(*args, **kwargs)
        finally:
            IDENT_TO_UUID.pop(thread.ident)

    return _orig_start_new_thread(wrapper, *args, **kwargs)


get_parent_uuid = UUIDS_TREE.get


def get_thread_parent(thread):
    """
    Retrieves parent thread for provided thread
    In case UUID table has parent but it's not valid anymore - returns DeadThread, so threads tree will be preserved

    :param thread:
    :return:
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
    import gevent.monkey
    gevent.monkey.saved['threading']['Thread'].parent = property(get_thread_parent)
    gevent.monkey.saved['threading']['Thread'].uuid = property(get_thread_uuid)
    IDENT_TO_UUID[main_thread_ident_before_patching] = get_thread_uuid()

    def iter_thread_frames():
        main_thread_frame = None
        for ident, frame in sys._current_frames().items():
            if IDENT_TO_UUID.get(ident) == MAIN_UUID:
                main_thread_frame = frame
                # the MainThread should be shown in it's "greenlet" version
                continue
            _logger.debug("thread - %s: %s", ident, threading._active.get(ident))
            yield ident, frame

        for thread in threading.enumerate():
            if not getattr(thread, '_greenlet', None):
                # some inbetween state, before greenlet started or after it died?...
                pass
            elif thread._greenlet.gr_frame:
                yield thread.ident, thread._greenlet.gr_frame
            else:
                # a thread with greenlet but without gr_frame will be fetched from sys._current_frames
                # If we switch to another greenlet by the time we get there we will get inconsistent dup of threads.
                # TODO - make best-effort attempt to show coherent thread dump
                yield thread.ident, main_thread_frame

else:
    def iter_thread_frames():
        yield from sys._current_frames().items()


this_module = import_module(__name__)
_BOOTSTRAPPERS = {threading, this_module}


def get_thread_tree(including_this=True):
    from .logging import THREAD_LOGGING_CONTEXT
    from .bunch import Bunch

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
        stacks[thread_ident] = formatted, time.time()

    def add_thread(parent_thread, parent):
        for thread in sorted(tree[parent_thread], key=lambda thread: thread.name):
            ident = thread.ident or 0
            stack, ts = stacks.get(ident, ("", 0))
            context = contexts.get(ident, {})
            context_line = ", ".join("%s: %s" % (k, context[k]) for k in "host context".split() if context.get(k))

            this = Bunch(
                name=thread.name,
                daemon="[D]" if getattr(thread, "daemon", False) else "",
                ident=ident,
                context_line="({})".format(context_line) if context_line else "",
                stack=stack,
                timestamp=ts,
                children=[],
                )
            parent.children.append(this)
            if thread in tree:
                add_thread(thread, this)
        return parent

    return add_thread(None, Bunch(children=[]))


def get_thread_stacks(including_this=True):

    stack_tree = get_thread_tree(including_this=including_this)

    def write_thread(parent):
        for branch in parent.children:
            with buff.indent('Thread{daemon}: {name} ({ident:X})', **branch):
                ts = time.strftime("%H:%M:%S", time.localtime(branch.timestamp))
                buff.write("{}  {}", ts, branch.context_line)
                for line in branch.stack.splitlines():
                    buff.write(line)
                write_thread(branch)

    from .humanize import IndentableTextBuffer
    buff = IndentableTextBuffer("Current Running Threads")
    write_thread(stack_tree)

    return buff


def watch_threads(interval):
    from easypy.resilience import resilient
    from easypy.concurrency import concurrent

    cmdline = " ".join(sys.argv)
    _logger = logging.getLogger(__name__)
    _threads_logger = logging.getLogger('threads')

    last_threads = set()

    @resilient.warning
    def dump_threads():
        nonlocal last_threads

        with _logger.indented('getting thread tree', level=logging.DEBUG):
            tree = get_thread_tree(including_this=False)

        with _logger.indented('logging threads to yaml', level=logging.DEBUG):
            _threads_logger.debug("threads", extra=dict(cmdline=cmdline, tree=tree.to_dict()))

        with _logger.indented('creating current thread set', level=logging.DEBUG):
            current_threads = set()
            for thread in threading.enumerate():
                if thread.ident:
                    current_threads.add(thread.ident)

        new_threads = current_threads - last_threads
        closed_threads = last_threads - current_threads
        stringify = lambda idents: ", ".join("%X" % ident for ident in idents)

        if new_threads:
            _logger.debug("WHITE<<NEW>> threads (%s): %s", len(new_threads), stringify(new_threads))
        if closed_threads:
            _logger.debug("threads terminated (%s): %s", len(closed_threads), stringify(closed_threads))
        _logger.debug("total threads: %s", len(current_threads))

        last_threads = current_threads

    thread = concurrent(dump_threads, threadname="ThreadWatch", loop=True, sleep=interval, real_thread_no_greenlet=True)
    thread.start()
    _logger.info("threads watcher started")


class ThreadContexts():
    """
    A structure for storing arbitrary data per thread.
    Unlike threading.local, data stored here will be inherited into
    child-threads (threads spawned from threads).

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

    'counters': attributes named here get incremented each time, instead of overwritten:

        TC = ThreadContexts(counters=('i', 'j'))
        assert TC.i == TC.j == 0

        with TC(i=1):
            assert TC.i == 1
            assert TC.j == 0

            with TC(i=1, j=1):
                assert TC.i == 2
                assert TC.j == 1


    'stacks': attributes named here get pushed into a list, instead of overwritten:

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
