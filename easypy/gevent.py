try:
    import gevent
    from gevent.monkey import is_module_patched
except ImportError:
    def is_module_patched(*_, **__):
        return False


from logging import getLogger, DEBUG
import threading   # this will be reloaded after patching

from queue import Queue
import time
import sys
from functools import partial
from itertools import count


from easypy.humanize import format_thread_stack
from . import _set_thread_uuid, _set_main_uuid


_logger = getLogger('gevent')


main_thread_ident_before_patching = threading.main_thread().ident

HUB = None

HOGGING_TIMEOUT = 1


def apply_patch(hogging_detection=False, real_threads=1):
    _logger.info('applying gevent patch (%s real threads)', real_threads)

    # real_threads is 1 by default so it will be possible to run watch_threads concurrently
    if hogging_detection:
        real_threads += 1

    if real_threads:
        _RealThreadsPool(real_threads)

    _patch_module_locks()

    import gevent
    import gevent.monkey

    for m in ["easypy.threadtree", "easypy.concurrency"]:
        assert m not in sys.modules, "Must apply the gevent patch before importing %s" % m

    gevent.monkey.patch_all(Event=True, sys=True)

    _unpatch_logging_handlers_lock()

    global HUB
    HUB = gevent.get_hub()

    global threading
    import threading
    for thread in threading.enumerate():
        _set_thread_uuid(thread.ident)
    _set_main_uuid()  # the patched threading has a new ident for the main thread

    # this will declutter the thread dumps from gevent/greenlet frames
    from .threadtree import _BOOTSTRAPPERS
    import gevent, gevent.threading, gevent.greenlet
    _BOOTSTRAPPERS.update([gevent, gevent.threading, gevent.greenlet])

    if hogging_detection:
        import greenlet
        greenlet.settrace(lambda *args: _greenlet_trace_func(*args))
        defer_to_thread(detect_hogging, 'detect-hogging')


def _patch_module_locks():
    # gevent will not patch existing locks (including ModuleLocks) when it's not single threaded
    # our solution is to monkey patch the release method for ModuleLocks objects
    # we assume that patching is done early enough so no other locks are present

    import importlib
    _old_release = importlib._bootstrap._ModuleLock.release

    def _release(*args, **kw):
        lock = args[0]
        if lock.owner == main_thread_ident_before_patching:
            lock.owner = threading.main_thread().ident
        _old_release(*args, **kw)

    importlib._bootstrap._ModuleLock.release = _release


def _unpatch_logging_handlers_lock():
    # we dont want to use logger locks since those are used by both real thread and gevent greenlets
    # switching from one to the other will cause gevent hub to throw an exception
    import logging

    RLock = gevent.monkey.saved['threading']['_CRLock']

    for handler in logging._handlers.values():
        if handler.lock:
            handler.lock = RLock()

    def create_unpatched_lock_for_handler(handler):
        handler.lock = RLock()

    # patch future handlers
    logging.Handler.createLock = create_unpatched_lock_for_handler


def _greenlet_trace_func(event, args):
    pass


def detect_hogging():
    did_switch = True

    current_running_greenlet = HUB

    def mark_switch(event, args):
        nonlocal did_switch
        nonlocal current_running_greenlet
        if event != 'switch':
            return
        did_switch = True
        current_running_greenlet = args[1]  # args = [origin_greenlet , target_greenlet

    global _greenlet_trace_func
    _greenlet_trace_func = mark_switch

    current_blocker_time = 0
    last_warning_time = 0

    while True:
        non_gevent_sleep(HOGGING_TIMEOUT)
        if did_switch:
            # all good
            pass
        elif current_running_greenlet == HUB:
            # it's ok for the hub to block if all greenlet wait on async io
            pass
        else:
            current_blocker_time += HOGGING_TIMEOUT
            if current_blocker_time < last_warning_time * 2:
                continue  # dont dump too much warnings - decay exponentialy until exploding after FAIL_BLOCK_TIME_SEC
            for thread in threading.enumerate():
                if getattr(thread, '_greenlet', None) == current_running_greenlet:
                    _logger.info('RED<<greentlet hogger detected (%s seconds):>>', current_blocker_time)
                    _logger.debug('thread stuck: %s', thread)
                    break
            else:
                _logger.info('RED<<unknown greentlet hogger detected (%s seconds):>>', current_blocker_time)
                _logger.debug('greenlet stuck (no corresponding thread found): %s', current_running_greenlet)
                _logger.debug('hub is: %s', HUB)
            _logger.debug("Stack:\n%s", format_thread_stack(sys._current_frames()[main_thread_ident_before_patching]))
            last_warning_time = current_blocker_time
            continue

        current_blocker_time = 0
        last_warning_time = 0
        did_switch = False


def non_gevent_sleep(timeout):
    try:
        gevent.monkey.saved['time']['sleep'](timeout)
    except KeyError:
        time.sleep(timeout)


class _RealThreadsPool():
    # We use this object to manage a pool of real OS threads, alongside the gevent thread and its greenlets

    POOL = None
    _job_id = count()

    def __init__(self, pool_size):
        assert not self.__class__.POOL, "Can't create more than one: %s" % self.__class__
        self.__class__.POOL = self
        self.active_jobs = set()
        self.pool_size = pool_size
        self.jobs_queue = Queue()

        def work():
            while True:
                func, name, job_id, parent_uuid = self.jobs_queue.get()
                _set_thread_uuid(threading.get_ident(), parent_uuid)
                _logger.debug('Starting job in real thread: %s', name or "<anonymous>")
                func()
                self.active_jobs.remove(job_id)
                _logger.debug('ready for the next job')

        for i in range(self.pool_size):
            name = 'real-thread-%s' % i
            thread = threading.Thread(target=work, name=name, daemon=True)
            thread.start()

    def _get_next_job_id(self):
        return next(self._job_id)

    def send_job(self, func, threadname=None):
        from .threadtree import get_thread_uuid
        parent_uuid = get_thread_uuid(threading.current_thread())
        job_id = self._get_next_job_id()
        self.jobs_queue.put((func, threadname, job_id, parent_uuid))
        self.active_jobs.add(job_id)
        return partial(self.join, job_id=job_id)

    def join(self, job_id):
        while job_id in self.active_jobs:
            non_gevent_sleep(0.5)


def defer_to_thread(func, threadname):
    return _RealThreadsPool.POOL.send_job(func, threadname)
