"""
This module monkey-patches python's (3.x) threading module so that we can find which thread spawned which.
This is useful in our contextualized logging, so that threads inherit logging context their 'parent'.
"""


import sys, traceback
from functools import lru_cache
from uuid import uuid4
import time
import logging
from copy import deepcopy
from contextlib import contextmanager

import _thread
import threading

from .bunch import Bunch
from .gevent import is_module_patched
from .collections import ilistify

UUIDS_TREE = {}
IDENT_TO_UUID = {}
UUID_TO_IDENT = {}


class Local(threading.local):
    def __create(self):
        self.uuid = uuid4()
        IDENT_TO_UUID[_thread.get_ident()] = self.uuid
        UUID_TO_IDENT[self.uuid] = _thread.get_ident()

    def get(self):
        if not hasattr(self, 'uuid'):
            self.__create()
        return self.uuid


local_uuid = Local()

_orig_start_new_thread = _thread.start_new_thread


def start_new_thread(target, *args, **kwargs):
    parent_uuid = local_uuid.get()

    def wrapper(*args, **kwargs):
        uuid = local_uuid.get()
        UUIDS_TREE[uuid] = parent_uuid
        return target(*args, **kwargs)

    return _orig_start_new_thread(wrapper, *args, **kwargs)


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
    @lru_cache(None)
    def get(cls, uuid):
        return cls(uuid)


get_parent_uuid = UUIDS_TREE.get


def get_thread_uuid(thread=None):
    if not thread:
        return local_uuid.get()
    return IDENT_TO_UUID.get(thread.ident)


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

# MainThread and DummyThread has to be patched separately because use of gevent
threading.Thread.parent = property(get_thread_parent)
threading.Thread.uuid = property(get_thread_uuid)

if is_module_patched("threading"):
    main_thread = threading.main_thread()
    main_thread.parent = threading._DummyThread.parent = property(get_thread_parent)
    main_thread.uuid = threading._DummyThread.uuid = property(get_thread_uuid)

    def iter_thread_frames():
        for thread in threading.enumerate():
            yield thread.ident, thread._greenlet.gr_frame if getattr(thread, '_greenlet', None) else None
else:
    def iter_thread_frames():
        yield from sys._current_frames().items()


def format_thread_stack(frame, after_module=threading):
    stack = traceback.extract_stack(frame)
    i = 0
    if after_module:
        # skip everything until after specified module
        for i, (fname, *_) in enumerate(stack):
            if fname == after_module.__file__:
                break
        for i, (fname, *row) in enumerate(stack[i:], i):
            if fname != after_module.__file__:
                break

    formatted = ""
    formatted += ''.join(traceback.format_list(stack[i:]))
    return formatted


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
            after_module = None if thread.ident in (current_ident, main_ident) else threading
            formatted = format_thread_stack(frame, after_module=after_module) if frame else ''
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

        tree = get_thread_tree(including_this=False)
        _threads_logger.debug("threads", extra=dict(cmdline=cmdline, tree=tree.to_dict()))

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

    thread = concurrent(dump_threads, threadname="ThreadWatch", loop=True, sleep=interval)
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
        self._context_data = {}
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
