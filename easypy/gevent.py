try:
    import gevent
    from gevent.monkey import is_module_patched
except ImportError:
    def is_module_patched(*_, **__):
        return False


import threading   # this will be reloaded after patching

import atexit
import time
import sys
import os


from ._multithreading_init import _set_thread_uuid, _set_main_uuid, _BOOTSTRAPPERS, get_thread_uuid
from . import yesno_to_bool

# can't use easypy's logging since this has to be run before everything,
# hence the name '_basic_logger', to remind that easypy features are not available
from logging import getLogger
_basic_logger = getLogger(name='gevent')


main_thread_ident_before_patching = threading.main_thread().ident

HUB = None

HOGGING_TIMEOUT = int(os.getenv('EASYPY_GEVENT_HOGGING_DETECTOR_INTERVAL', 0))
_HOGGING_DETECTION_RUNNING = False


def apply_patch(hogging_detection=None, real_threads=None):
    if real_threads is None:
        real_threads = int(os.getenv('EASYPY_GEVENT_REAL_THREADS', 1))
    if hogging_detection is None:
        hogging_detection = bool(HOGGING_TIMEOUT)

    _basic_logger.info('applying gevent patch (%s real threads)', real_threads)

    # real_threads is 1 by default so it will be possible to run watch_threads concurrently
    if hogging_detection:
        real_threads += 1

    import gevent
    import gevent.monkey

    for m in ["easypy.threadtree", "easypy.concurrency"]:
        assert m not in sys.modules, "Must apply the gevent patch before importing %s" % m

    gevent.monkey.patch_all(Event=True, sys=True)

    _patch_module_locks()
    _unpatch_logging_handlers_lock()

    global HUB
    HUB = gevent.get_hub()

    global threading
    import threading
    for thread in threading.enumerate():
        _set_thread_uuid(thread.ident)
    _set_main_uuid()  # the patched threading has a new ident for the main thread

    # this will declutter the thread dumps from gevent/greenlet frames
    import gevent, gevent.threading, gevent.greenlet
    _BOOTSTRAPPERS.update([gevent, gevent.threading, gevent.greenlet])

    if hogging_detection:
        import greenlet
        greenlet.settrace(lambda *args: _greenlet_trace_func(*args))
        global _HOGGING_DETECTION_RUNNING
        _HOGGING_DETECTION_RUNNING = True
        wait = defer_to_thread(detect_hogging, 'detect-hogging')

        @atexit.register
        def stop_detection():
            global _HOGGING_DETECTION_RUNNING
            _HOGGING_DETECTION_RUNNING = False
            wait()


def _patch_module_locks():
    # gevent will not patch existing locks (including ModuleLocks) when it's not single threaded
    # so we map the ownership of module locks to the greenlets that took over

    import importlib
    thread_greenlet_ident = {
        main_thread_ident_before_patching: threading.main_thread().ident
    }

    for ref in importlib._bootstrap._module_locks.values():
        lock = ref()
        lock.owner = thread_greenlet_ident.get(lock.owner, lock.owner)


def _unpatch_logging_handlers_lock():
    # we dont want to use logger locks since those are used by both real thread and gevent greenlets
    # switching from one to the other will cause gevent hub to throw an exception

    RLock = gevent.monkey.saved['threading']['_CRLock']

    def create_unpatched_lock_for_handler(handler):
        handler.lock = RLock()

    import logging
    # patch future handlers
    logging.Handler.createLock = create_unpatched_lock_for_handler
    for handler in logging._handlers.values():
        if handler.lock:
            handler.lock = RLock()

    try:
        import logbook.handlers
    except ImportError:
        pass
    else:
        # patch future handlers
        logbook.handlers.new_fine_grained_lock = RLock
        for handler in logbook.handlers.Handler.stack_manager.iter_context_objects():
            handler.lock = RLock()


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

    while _HOGGING_DETECTION_RUNNING:
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
                    _basic_logger.info('RED<<greenlet hogger detected (%s seconds):>>', current_blocker_time)
                    _basic_logger.debug('thread stuck: %s', thread)
                    break
            else:
                _basic_logger.info('RED<<unknown greenlet hogger detected (%s seconds):>>', current_blocker_time)
                _basic_logger.debug('greenlet stuck (no corresponding thread found): %s', current_running_greenlet)
                _basic_logger.debug('hub is: %s', HUB)
            # this is needed by `detect_hogging`, but we must'nt import it
            # there since it leads to a gevent/native-thread deadlock,
            # and can't import it at the top since thing must wait for
            # the gevent patching
            from easypy.humanize import format_thread_stack
            func = _basic_logger.debug if current_blocker_time < 5 * HOGGING_TIMEOUT else _basic_logger.info
            func("Stack:\n%s", format_thread_stack(sys._current_frames()[main_thread_ident_before_patching]))
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


def defer_to_thread(func, threadname):

    def run():
        _set_thread_uuid(threading.get_ident(), parent_uuid)
        _basic_logger.debug('Starting job in real thread: %s', threadname or "<anonymous>")
        gevent.spawn(func)  # running via gevent ensures we have a Hub
        gevent.wait()
        _basic_logger.debug('ready for the next job')

    parent_uuid = get_thread_uuid(threading.current_thread())
    pool = gevent.get_hub().threadpool
    return pool.spawn(run).wait
