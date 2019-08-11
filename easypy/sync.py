"""
This module is about synchronizing and coordinating events among concurrent activities.
"""


from contextlib import contextmanager, ExitStack
import sys
import threading
import time
import inspect
from functools import wraps
import re
import logging
import atexit
import signal
import os
from collections import Counter

import easypy._multithreading_init  # noqa
from .bunch import Bunch
from .gevent import is_module_patched
from .decorations import wrapper_decorator, parametrizeable_decorator
from .caching import locking_cache
from .exceptions import PException, TException
from .units import NEVER, MINUTE, HOUR
from .misc import Hex
from .humanize import time_duration  # due to interference with jrpc
from .misc import kwargs_resilient

_logger = logging.getLogger(__name__)
_verbose_logger = logging.getLogger('%s.locks' % __name__)  # logger for less important logs of RWLock.

IS_A_TTY = sys.stdout.isatty()


class TimeoutException(PException, TimeoutError):
    pass


class PredicateNotSatisfied(TimeoutException):
    # use this exception with 'wait', to indicate predicate is not satisfied
    # and allow it to raise a more informative exception
    pass


class LockLeaseExpired(TException):
    template = "Lock Lease Expired - thread is holding this lock for too long"


class ProcessExiting(TException):
    template = "Aborting thread - process is exiting"


class TimebombExpired(TException):
    template = "Timebomb Expired - process killed itself"
    exit_with_code = 234


_exiting = False  # we use this to break out of lock-acquisition loops


@atexit.register
def break_locks():
    global _exiting
    _exiting = True


def _check_exiting():
    if _exiting:
        raise ProcessExiting()


class TerminationSignal(TException):
    template = "Process got a termination signal: {_signal}"


class NotMainThread(TException):
    template = "Binding must be invoked from main thread"


class SignalAlreadyBound(TException):
    template = "Signal already bound to another signal handler(s)"


class LastErrorEmpty(TException):
    template = "Signal caught, but no error to raise"


LAST_ERROR = None
REGISTERED_SIGNAL = None


class NotInitialized(TException):
    template = "Signal type not initialized, must use bind_to_subthread_exceptions in the main thread"


def async_raise_in_main_thread(exc, use_concurrent_loop=True):
    """
    Uses a unix signal to raise an exception to be raised in the main thread.
    """

    from plumbum import local
    pid = os.getpid()
    if not REGISTERED_SIGNAL:
        raise NotInitialized()

    # sometimes the signal isn't caught by the main-thread, so we should try a few times (WEKAPP-14543)
    def do_signal(raised_exc):
        global LAST_ERROR
        if LAST_ERROR is not raised_exc:
            _logger.debug("MainThread took the exception - we're done here")
            if use_concurrent_loop:
                raiser.stop()
            return

        _logger.info("Raising %s in main thread", type(LAST_ERROR))
        local.cmd.kill("-%d" % REGISTERED_SIGNAL, pid)

    if use_concurrent_loop:
        from .concurrency import concurrent
        raiser = concurrent(do_signal, raised_exc=exc, loop=True, sleep=30, daemon=True, throw=False)
        raiser.start()
    else:
        do_signal(exc)



if is_module_patched("threading"):
    import gevent
    def _rimt(exc):
        _logger.info('YELLOW<<killing main thread greenlet>>')
        main_thread_greenlet = threading.main_thread()._greenlet
        orig_throw = main_thread_greenlet.throw

        # we must override "throw" method so exception will be raised with the original traceback
        def throw(*args):
            if len(args) == 1:
                ex = args[0]
                return orig_throw(ex.__class__, ex, ex.__traceback__)
            return orig_throw(*args)
        main_thread_greenlet.throw = throw
        gevent.kill(main_thread_greenlet, exc)
        _logger.debug('exiting the thread that failed')
        raise exc
else:
    _rimt = async_raise_in_main_thread


# must be invoked in main thread in "geventless" runs in order for raise_in_main_thread to work
def initialize_exception_listener():
    global REGISTERED_SIGNAL
    if REGISTERED_SIGNAL:
        # already registered
        return

    if threading.current_thread() is not threading.main_thread():
        raise NotMainThread()

    def handle_signal(sig, stack):
        global LAST_ERROR
        error = LAST_ERROR
        LAST_ERROR = None
        if error:
            raise error
        raise LastErrorEmpty(signal=sig)

    custom_signal = signal.SIGUSR1
    if signal.getsignal(custom_signal) in (signal.SIG_DFL, signal.SIG_IGN):  # check if signal is already trapped
        signal.signal(custom_signal, handle_signal)
        REGISTERED_SIGNAL = custom_signal
    else:
        raise SignalAlreadyBound(signal=custom_signal)


@contextmanager
def raise_in_main_thread(exception_type=Exception):

    try:
        yield
    except ProcessExiting:
        # this exception is meant to stay within the thread
        raise
    except exception_type as exc:
        if threading.current_thread() is threading.main_thread():
            raise
        exc._raised_asynchronously = True

        global LAST_ERROR
        if LAST_ERROR:
            _logger.warning("a different error (%s) is pending - skipping", type(LAST_ERROR))
            raise
        LAST_ERROR = exc
        _rimt(exc)


def initialize_termination_listener(sig=signal.SIGTERM, _registered=[]):
    if _registered:
        # already registered
        return
    _registered.append(True)

    def handle_signal(sig, stack):
        _logger.error("RED<<SIGNAL %s RECEIVED>>", sig)
        raise TerminationSignal(_signal=sig)

    if signal.getsignal(sig) in (signal.SIG_DFL, signal.SIG_IGN):  # check if signal is already trapped
        _logger.info("Listening to signal %s", sig)
        signal.signal(sig, handle_signal)
    else:
        raise SignalAlreadyBound(signal=sig)


def kill_subprocesses():
    from plumbum import local
    pid = os.getpid()
    return local.cmd.pkill['-HUP', '-P', pid].run(retcode=None)


def kill_this_process(graceful=False):
    from plumbum import local
    pid = os.getpid()
    if graceful:
        flag = '-HUP'
    else:
        flag = '-9'
    local.cmd.kill(flag, pid)


class Timebomb(object):

    def __init__(self, timeout, alert_interval=None, quiet=False):
        self.fuse = threading.Event()  # use this to cancel the timebomb
        self.timeout = timeout
        self.alert_interval = alert_interval
        self.quiet = quiet

    def __enter__(self):
        return self.start()

    def __exit__(self, *args):
        self.cancel()

    def start(self):
        from .concurrency import concurrent
        self.t = concurrent(self.wait_and_kill, daemon=True, threadname="Timebomb(%s)" % self.timeout)
        self.t.start()
        return self

    def cancel(self):
        self.fuse.set()

    @raise_in_main_thread()
    def wait_and_kill(self):
        timer = Timer(expiration=self.timeout)
        if not self.quiet:
            _logger.info("Timebomb set - this process will YELLOW<<self-destruct>> in RED<<%r>>...", timer.remain)
        while not timer.expired:
            if self.alert_interval:
                _logger.info("Time Elapsed: MAGENTA<<%r>>", timer.elapsed)
            log_level = logging.WARNING if timer.remain < 5 * MINUTE else logging.DEBUG
            _logger.log(log_level, "This process will YELLOW<<self-destruct>> in RED<<%r>>...", timer.remain)
            if self.fuse.wait(min(self.alert_interval or 60, timer.remain)):
                _logger.info("Timebomb cancelled")
                return
        _logger.warning("RED<< 💣 Timebomb Expired! 💣 >>")
        with _logger.indented("Killing children"):
            kill_subprocesses()
        with _logger.indented("Committing suicide"):
            kill_this_process(graceful=False)
            raise Exception("Timebomb Expired")


def set_timebomb(timeout, alert_interval=None):
    return Timebomb(timeout=timeout, alert_interval=alert_interval).start()


@wrapper_decorator
def shared_contextmanager(func):

    @locking_cache
    def inner(*args, **kwargs):

        class CtxManager():
            def __init__(self):
                self.count = 0
                self.func_cm = contextmanager(func)
                self._lock = threading.RLock()

            def __enter__(self):
                with self._lock:
                    if self.count == 0:
                        self.ctm = self.func_cm(*args, **kwargs)
                        self.obj = self.ctm.__enter__()
                    self.count += 1
                return self.obj

            def __exit__(self, *args):
                with self._lock:
                    self.count -= 1
                    if self.count > 0:
                        return
                    self.ctm.__exit__(*sys.exc_info())
                    del self.ctm
                    del self.obj

        return CtxManager()

    return inner


class TagAlongThread(object):

    def __init__(self, func, name, minimal_sleep=0):
        self._func = func
        self.minimal_sleep = minimal_sleep

        self._iteration_trigger = threading.Event()

        self._iterating = threading.Event()
        self._not_iterating = threading.Event()
        self._not_iterating.set()

        self._last_exception = None
        self._last_result = None

        self.__alive = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name=name)
        self._thread.start()

    def _loop(self):
        while self.__alive:
            self._iteration_trigger.wait()
            self._iteration_trigger.clear()

            # Mark that we are now iterating
            self._not_iterating.clear()
            self._iterating.set()

            try:
                self._last_exception, self._last_result = None, self._func()
            except Exception as e:
                self._last_exception = e

            # Mark that we are no longer iterating
            self._iterating.clear()
            self._not_iterating.set()

            time.sleep(self.minimal_sleep)

        # Set all events so nothing will get blocked
        self._iteration_trigger.set()
        self._iterating.set()
        self._not_iterating.set()

    def __repr__(self):
        return 'TagAlongThread<%s>' % (self._thread.name,)

    def __call__(self):
        assert self.__alive, '%s is dead' % self
        # We can't use an iteration that's already started - maybe it's already at a too advanced stage?
        if self._iterating.is_set():
            self._not_iterating.wait()

        self._iteration_trigger.set()  # Signal that we want an iteration

        while not self._iterating.wait(1):  # Wait until an iteration starts
            # It is possible that we missed the loop and _iterating was already
            # cleared. If this is the case, _not_iterating will not be set -
            # and we can use it as a signal to stop waiting for iteration.
            if self._not_iterating.is_set():
                break
        else:
            self._not_iterating.wait()  # Wait until it finishes

        # To avoid races, copy last exception and result to local variables
        last_exception, last_result = self._last_exception, self._last_result
        if last_exception:
            raise last_exception
        else:
            return last_result

    def _kill(self, wait=True):
        self.__alive = False
        self._iteration_trigger.set()
        if wait:
            self._thread.join()


class SynchronizationCoordinatorWrongWait(TException):
    template = "Task is waiting on {this_file}:{this_line} instead of {others_file}:{others_line}"


class SynchronizationCoordinator(object):
    """
    Synchronization helper for functions that run concurrently::

        sync = SynchronizationCoordinator(5)

        def foo(a):
            sync.wait_for_everyone()
            sync.collect_and_call_once(a, lambda a_values: print(a))

        MultiObject(range(5)).call(foo)

    When MultiObject/concurrent_map/sync runs a function with a ``_sync=SYNC``
    argument, it will replace it with a proper SynchronizationCoordinator instance::

        def foo(a, _sync=SYNC):
            _sync.wait_for_everyone()
            _sync.collect_and_call_once(a, lambda a_values: print(a))

        MultiObject(range(5)).call(foo)
    """

    def __init__(self, num_participants):
        self.num_participants = num_participants
        self._reset_barrier()
        self._lock = threading.Lock()
        self._call_once_collected_param = []
        self._call_once_function = None
        self._call_once_result = None
        self._call_once_raised_exception = False

        self._wait_context = None

    def _reset_barrier(self):
        self.barrier = threading.Barrier(self.num_participants, action=self._post_barrier_action)

    def _post_barrier_action(self):
        if self.num_participants != self.barrier.parties:
            self._reset_barrier()
        self._wait_context = None

        if self._call_once_function:
            call_once_function, self._call_once_function = self._call_once_function, None
            collected_param, self._call_once_collected_param = self._call_once_collected_param, []

            try:
                self._call_once_result = call_once_function(collected_param)
                self._call_once_raised_exception = False
            except BaseException as e:
                self._call_once_result = e
                self._call_once_raised_exception = True

    def wait_for_everyone(self, timeout=HOUR):
        """
        Block until all threads that participate in the synchronization coordinator reach this point.

        Fail if one of the threads is waiting at a different point::

            def foo(a, _sync=SYNC):
                sleep(a)
                # Each thread will reach this point at a different time
                _sync.wait_for_everyone()
                # All threads will reach this point together

            MultiObject(range(5)).call(foo)
        """
        self._verify_waiting_on_same_line()
        self.barrier.wait(timeout=timeout)

    def abandon(self):
        """
        Stop participating in this synchronization coordinator.

        Note: when using with MultiObject/concurrent_map/asynchronous and _sync=SYNC, this
        is called automatically when a thread terminates on return or on exception.
        """
        with self._lock:
            self.num_participants -= 1
        self.barrier.wait()

    def collect_and_call_once(self, param, func, *, timeout=HOUR):
        """
        Call a function from one thread, with parameters collected from all threads::

            def foo(a, _sync=SYNC):
                result = _sync.collect_and_call_once(a, lambda a_values: set(a_values))
                assert result == {0, 1, 2, 3, 4}

            MultiObject(range(5)).call(foo)
        """
        self._verify_waiting_on_same_line()

        self._call_once_collected_param.append(param)
        self._call_once_function = func  # this will be set multiple times - but there is no race so that's OK

        self.barrier.wait(timeout=timeout)

        if self._call_once_raised_exception:
            raise self._call_once_result
        else:
            return self._call_once_result

    def _verify_waiting_on_same_line(self):
        frame = inspect.currentframe().f_back.f_back
        wait_context = (frame.f_code, frame.f_lineno, frame.f_lasti)

        existing_wait_context = self._wait_context
        if existing_wait_context is None:
            with self._lock:
                # Check again inside the lock, in case it was changed
                existing_wait_context = self._wait_context
                if existing_wait_context is None:
                    self._wait_context = wait_context
        if existing_wait_context is not None:  # could have changed inside the lock
            if wait_context != existing_wait_context:
                self.barrier.abort()
                raise SynchronizationCoordinatorWrongWait(
                    this_file=wait_context[0].co_filename,
                    this_line=wait_context[1],
                    others_file=existing_wait_context[0].co_filename,
                    others_line=existing_wait_context[1])

    def _abandon_when_done(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                if hasattr(result, '__enter__') and hasattr(result, '__exit__'):
                    @contextmanager
                    def wrapper_cm():
                        try:
                            with result as yielded_value:
                                self.wait_for_everyone()  # see https://github.com/weka-io/easypy/issues/150
                                yield yielded_value
                        finally:
                            self.abandon()
                    return wrapper_cm()
                else:
                    self.abandon()
                    return result
            except:
                self.abandon()
                raise
        return wrapper


class SYNC(SynchronizationCoordinator):
    """Mimic ``SynchronizationCoordinator`` for running in single thread."""

    def __init__(self):
        pass

    def wait_for_everyone(self):
        pass
    wait_for_everyone.__doc__ = SynchronizationCoordinator.wait_for_everyone.__doc__

    def abandon(self):
        pass
    abandon.__doc__ = SynchronizationCoordinator.abandon.__doc__

    def collect_and_call_once(self, param, func):
        return func([param])
    collect_and_call_once.__doc__ = SynchronizationCoordinator.collect_and_call_once.__doc__


SYNC = SYNC()


def _get_my_ident():
    return Hex(threading.current_thread().ident)


class LoggedRLock():
    """
    Like ``RLock``, but more logging friendly.

    :param name: give it a name, so it's identifiable in the logs
    :param log_interval: the interval between log messages
    :param lease_expiration: throw an exception if the lock is held for more than this duration
    """

    # we could inherit from this and support other types, but that'll require changes in the repr
    LockType = threading.RLock

    __slots__ = ("_lock", "_name", "_lease_expiration", "_lease_timer", "_log_interval", "_get_data")
    _RE_OWNER = re.compile(r".*owner=(\d+) count=(\d+).*")
    _MIN_TIME_FOR_LOGGING = 10

    def __init__(self, name=None, log_interval=15, lease_expiration=NEVER):
        self._lock = self.__class__.LockType()
        self._name = name or '{}-{:X}'.format(self.LockType.__name__, id(self))
        self._lease_expiration = lease_expiration
        self._lease_timer = None
        self._log_interval = log_interval

        # we want to support both the gevent and builtin lock
        try:
            self._lock._owner
        except AttributeError:
            def _get_data():
                return tuple(map(int, self._RE_OWNER.match(repr(self._lock)).groups()))
        else:
            def _get_data():
                return self._lock._owner, self._lock._count
        self._get_data = _get_data

    def __repr__(self):
        owner, count = self._get_data()
        try:
            owner = threading._active[owner].name
        except KeyError:
            pass
        if owner:
            return "<{}, owned by <{}>x{} for {}>".format(self._name, owner, count, self._lease_timer.elapsed)
        else:
            return "<{}, unowned>".format(self._name)

    def _acquired(self, lease_expiration, should_log=False):
        # we don't want to replace the lease timer, so not to effectively extend the original lease
        self._lease_timer = self._lease_timer or Timer(expiration=lease_expiration or self._lease_expiration)
        if should_log:
            _logger.debug("%s - acquired", self)

    def acquire(self, blocking=True, timeout=-1, lease_expiration=None):
        if not blocking:
            ret = self._lock.acquire(blocking=False)
            # touch it once, so we don't hit a race since it occurs outside of the lock acquisition
            lease_timer = self._lease_timer
            if ret:
                self._acquired(lease_expiration)
            elif lease_timer and lease_timer.expired:
                raise LockLeaseExpired(lock=self)
            return ret

        # this timer implements the 'timeout' parameter
        acquisition_timer = Timer(expiration=NEVER if timeout < 0 else timeout)
        while not acquisition_timer.expired:

            # the timeout on actually acquiring this lock is the minimum of:
            # 1. the time remaining on the acquisition timer, set by the 'timeout' param
            # 2. the logging interval - the minimal frequency for logging while the lock is awaited
            # 3. the time remaining on the lease timer, which would raise if expired
            timeout = min(acquisition_timer.remain, self._log_interval)
            # touch it once, so we don't hit a race since it occurs outside of the lock acquisition
            lease_timer = self._lease_timer
            if lease_timer:
                timeout = min(lease_timer.remain, timeout)

            if self._lock.acquire(blocking=True, timeout=timeout):
                self._acquired(lease_expiration, should_log=acquisition_timer.elapsed > self._MIN_TIME_FOR_LOGGING)
                return True

            # touch it once, so we don't hit a race since it occurs outside of the lock acquisition
            lease_timer = self._lease_timer
            if lease_timer and lease_timer.expired:
                raise LockLeaseExpired(lock=self)

            _logger.debug("%s - waiting...", self)

    def release(self, *args):
        _, count = self._get_data()
        if count == 1:
            # we're last: clear the timer before releasing the lock!
            if self._lease_timer.elapsed > self._MIN_TIME_FOR_LOGGING:
                _logger.debug("%s - releasing...", self)
            self._lease_timer = None
        self._lock.release()

    __exit__ = release
    __enter__ = acquire


class RWLock(object):
    """
    Read-Write Lock: allows locking exclusively and non-exclusively::

        rwl = RWLock()

        with rwl:
            # other can acquire this lock, but not exclusively

        with rwl.exclusive():
            # no one can acquire this lock - we are alone here

    """

    def __init__(self, name=None):
        self.lock = threading.RLock()
        self.cond = threading.Condition(self.lock)
        self.owners = Counter()
        self.name = name or '{}-{:X}'.format(self.__class__.__name__, id(self.lock))
        self._lease_timer = None

    def __repr__(self):
        owners = ", ".join(map(str, sorted(self.owners.keys())))
        lease_timer = self._lease_timer  # touch once to avoid races
        if lease_timer:
            mode = "exclusively ({})".format(lease_timer.elapsed)
        else:
            mode = "non-exclusively"
        return "<{}, owned by <{}> {}>".format(self.name, owners, mode)

    @property
    def owner_count(self):
        return sum(self.owners.values())

    def __call__(self):
        return self

    def __enter__(self):
        while not self.cond.acquire(timeout=15):
            _logger.debug("%s - waiting...", self)

        try:
            self.owners[_get_my_ident()] += 1
            _verbose_logger.debug("%s - acquired (non-exclusively)", self)
            return self
        finally:
            self.cond.release()

    def __exit__(self, *args):
        while not self.cond.acquire(timeout=15):
            _logger.debug("%s - waiting...", self)

        try:
            my_ident = _get_my_ident()
            self.owners[my_ident] -= 1
            if not self.owners[my_ident]:
                self.owners.pop(my_ident)  # don't inflate the soft lock keys with threads that does not own it
            self.cond.notify()
            _verbose_logger.debug("%s - released (non-exclusive)", self)
        finally:
            self.cond.release()

    @contextmanager
    def exclusive(self, need_to_wait_message=None):
        while not self.cond.acquire(timeout=15):
            _logger.debug("%s - waiting...", self)

        # wait until this thread is the sole owner of this lock
        while not self.cond.wait_for(lambda: self.owner_count == self.owners[_get_my_ident()], timeout=15):
            _check_exiting()
            if need_to_wait_message:
                _logger.info(need_to_wait_message)
                need_to_wait_message = None  # only print it once
            _logger.debug("%s - waiting (for exclusivity)...", self)
        my_ident = _get_my_ident()
        self.owners[my_ident] += 1
        self._lease_timer = Timer()
        _verbose_logger.debug("%s - acquired (exclusively)", self)
        try:
            yield
        finally:
            _verbose_logger.debug('%s - releasing (exclusive)', self)
            self._lease_timer = None
            self.owners[my_ident] -= 1
            if not self.owners[my_ident]:
                self.owners.pop(my_ident)  # don't inflate the soft lock keys with threads that does not own it
            self.cond.notify()
            self.cond.release()


SoftLock = RWLock


@parametrizeable_decorator
def skip_if_locked(func=None, lock=None, default=None):
    if not lock:
        lock = threading.RLock()

    def inner(*args, **kwargs):
        if not lock.acquire(blocking=False):
            _logger.debug("lock acquired - skipped %s", func)
            return default
        try:
            return func(*args, **kwargs)
        finally:
            lock.release()
    return inner


@parametrizeable_decorator
def with_my_lock(method, lock_attribute="_lock"):

    def inner(self, *args, **kwargs):
        ctx = getattr(self, lock_attribute)
        if callable(ctx):
            ctx = ctx()
        with ctx:
            return method(self, *args, **kwargs)

    return inner


@parametrizeable_decorator
def synchronized(func, lock=None):
    if not lock:
        lock = threading.RLock()

    def inner(*args, **kwargs):
        with lock:
            return func(*args, **kwargs)

    return inner


class SynchronizedSingleton(type):
    _instances = {}

    @synchronized
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)

        return cls._instances[cls]

    @synchronized
    def get_instance(cls):
        return cls._instances.get(cls)


class LoggedCondition():
    """
    Like Condition, but easier to use and more logging friendly

    :param name: give it a name, so it's identifiable in the logs
    :param log_interval: the interval between log messages

    Unlike threading.condition, .acquire() and .release() are not needed here.
    Just use .wait_for() to wait for a predicate and perform the
    condition-changing statements inside a .notifying_all() context::

        some_flag = False
        cond = Condition('some flag cond')

        # Wait for condition:
        cond.wait_for(lambda: some_flag, 'some flag become True')

        # Trigger the condition:
        with cond.notifying_all('Setting some flag to true'):
            some_flag = True
    """

    ConditionType = threading.Condition

    __slots__ = ("_cond", "_name", "_log_interval")

    def __init__(self, name=None, log_interval=15):
        self._cond = self.__class__.ConditionType()
        self._name = name or '{}-{:X}'.format(self.ConditionType.__name__, id(self))
        self._log_interval = log_interval

    def __repr__(self):
        return '<{}>'.format(self._name)

    @contextmanager
    def _acquired_for(self, msg, *args):
        while not self._cond.acquire(timeout=self._log_interval):
            _logger.debug('%s - waiting to be acquired for ' + msg, self, *args)
        try:
            yield
        finally:
            self._cond.release()

    @contextmanager
    def notifying_all(self, msg, *args):
        """
        Acquire the condition lock for the context, and notify all waiters afterward.

        :param msg: Message to print to the DEBUG log after performing the command
        :param args: Format arguments for msg

        Users should run the command that triggers the conditions inside this context manager.
        """
        with self._acquired_for('performing a %s notifying all waiters' % msg, *args):
            yield
            _logger.debug('%s - performed: ' + msg, self, *args)
            self._cond.notifyAll()

    @contextmanager
    def __wait_for_impl(self, pred, msg, *args, timeout=None):
        timer = Timer(expiration=timeout)

        def timeout_for_condition():
            remain = timer.remain
            if remain:
                return min(remain, self._log_interval)
            else:
                return self._log_interval

        with self._acquired_for('checking ' + msg, *args):
            while not self._cond.wait_for(pred, timeout=timeout_for_condition()):
                if timer.expired:
                    # NOTE: without a timeout we will never get here
                    if pred():  # Try one last time, to make sure the last check was not (possibly too long) before the timeout
                        return
                    raise TimeoutException('{condition} timed out after {duration} waiting for {msg}',
                                           condition=self, msg=msg % args, duration=timer.elapsed)
                _logger.debug('%s - waiting for ' + msg, self, *args)
            yield

    def wait_for(self, pred, msg, *args, timeout=None):
        """
        Wait for a predicate. Only check it when notified.

        :param msg: Message to print to the DEBUG log while waiting for the predicate
        :param args: Format arguments for msg
        :param timeout: Maximal time to wait
        """
        with self.__wait_for_impl(pred, msg, *args, timeout=timeout):
            pass

    @contextmanager
    def waited_for(self, pred, msg, *args, timeout=None):
        """
        Wait for a predicate, keep the condition lock for the context, and notify all other waiters afterward.

        :param msg: Message to print to the DEBUG log while waiting for the predicate
        :param args: Format arguments for msg
        :param timeout: Maximal time to wait

        The code inside the context should be used for altering state other waiters are waiting for.
        """
        with self.__wait_for_impl(pred, msg, *args, timeout=timeout):
            yield
            self._cond.notifyAll()

    @property
    def lock(self):
        """
        Use the underlying lock without notifying the waiters.
        """

        return self._cond._lock


# cache result only when predicate succeeds
class CachingPredicate():
    def __init__(self, pred):
        self.pred = pred

    def __call__(self, *args, **kwargs):
        try:
            return self.result
        except AttributeError:
            pass
        ret = self.pred(*args, **kwargs)
        if ret in (False, None):
            return ret
        self.result = ret
        return self.result


def make_multipred(preds):
    preds = list(map(CachingPredicate, preds))

    def pred(*args, **kwargs):
        results = [pred(*args, **kwargs) for pred in preds]
        if all(results):
            return results
    return pred


def iter_wait(
        timeout, pred=None, sleep=0.5, message=None,
        progressbar=True, throw=True, allow_interruption=False, caption=None,
        log_interval=10 * MINUTE, log_level=logging.DEBUG):

    # Calling wait() with a predicate and no message is very not informative
    # (throw=False or message=False disables this behavior)
    if message is False:
        message = None
    elif throw and pred and not message:
        raise Exception(
            "Function iter_wait()'s parameter `message` is required if "
            "`pred` is passed",
        )

    if timeout is None:
        msg = "Waiting indefinitely%s"
    else:
        msg = "Waiting%%s up to %s" % time_duration(timeout)

    if message is None:
        if caption:
            message = "Waiting %s timed out after {duration:.1f} seconds" % (caption,)
        elif pred:
            message = "Waiting on predicate (%s) timed out after {duration:.1f} seconds" % (pred,)
        else:
            message = "Timed out after {duration:.1f} seconds"

    if pred:
        pred_decorator = kwargs_resilient(negligible=['is_final_attempt'])
        if hasattr(pred, "__iter__"):
            pred = make_multipred(map(pred_decorator, pred))
        else:
            pred = pred_decorator(pred)
        if not caption:
            caption = "on predicate (%s)" % pred
    else:
        pred = lambda **kwargs: False
        throw = False

    if caption:
        msg %= " %s" % (caption,)
    else:
        msg %= ""

    if isinstance(sleep, tuple):
        data = list(sleep)  # can't use nonlocal since this module is indirectly used in python2

        def sleep():
            cur, mx = data
            try:
                return cur
            finally:
                data[0] = min(mx, cur * 1.5)

    if not IS_A_TTY:
        # can't interrupt
        allow_interruption = False
        progressbar = False

    if progressbar and threading.current_thread() is not threading.main_thread():
        # prevent clutter
        progressbar = False

    if allow_interruption:
        msg += " (hit <ESC> to continue)"

    l_timer = Timer(expiration=timeout)
    log_timer = Timer(expiration=log_interval)

    with ExitStack() as stack:
        if progressbar:
            from .logging import PROGRESS_BAR
            pr = stack.enter_context(PROGRESS_BAR())
            pr.set_message(msg)

        while True:
            s_timer = Timer()
            expired = l_timer.expired
            last_exc = None
            try:
                ret = pred(is_final_attempt=bool(expired))
            except PredicateNotSatisfied as _exc:
                if getattr(_exc, "duration", 0):
                    # this exception was raised by a nested 'wait' call - don't swallow it!
                    raise
                if log_timer.expired:
                    log_timer.reset()
                    _logger.log(log_level, 'Still waiting after %r: %s', l_timer.elapsed, _exc.message)
                last_exc = _exc
                ret = None
            else:
                if ret not in (None, False):
                    yield ret
                    return
            if expired:
                duration = l_timer.stop()
                start_time = l_timer.start_time
                if throw:
                    if last_exc:
                        last_exc.add_params(duration=duration, start_time=start_time)
                        raise last_exc
                    if callable(message):
                        message = message()
                    raise TimeoutException(message, duration=duration, start_time=start_time)
                yield None
                return
            yield l_timer.remain
            sleep_for = sleep() if callable(sleep) else sleep
            if allow_interruption:
                from termenu.keyboard import keyboard_listener
                timer = Timer(expiration=sleep_for - s_timer.elapsed)
                for key in keyboard_listener(heartbeat=0.25):
                    if key == "esc":
                        yield None
                        return
                    if key == "enter":
                        break
                    if timer.expired:
                        break
            else:
                s_timeout = max(0, sleep_for - s_timer.elapsed)
                if l_timer.expiration:
                    s_timeout = min(l_timer.remain, s_timeout)
                time.sleep(s_timeout)


@wraps(iter_wait)
def wait(*args, **kwargs):
    """
    Wait until ``pred`` returns a useful value (see below), or until ``timeout`` passes.

    :param timeout: how long to wait for ``pred`` to get satisfied. if ``None`` waits
        indefinitely.
    :param pred: callable that checks the condition to wait upon.
        It can return ``None`` or ``False`` to indicate that the predicate has not been satisfied.
        Any other value will end the wait and that value will be returned from the wait function.
        The predicate can also raise a subclass of PredicateNotSatisfied. The exception will be raised
        if the timeout expires, instead of a TimeoutException.
        ``pred`` can be a list of predicates, and ``wait`` will wait for all of them to be satisfied.
        Note that once a predicate is satisfied, it will not be called again.
        If no ``pred`` is provided, ``wait`` behaves like ``sleep``.
    :param sleep: the number of seconds to sleep between calls to ``pred``.
        it can be a callable, or a ``(first, max)`` tuple, in which case the sleep duration will grow
        exponentially from ``first`` up to ``max``.
    :param message: message to use for a TimeoutException. can be a callable. To encourage the use of
        informative TimeoutException messages, the user must provide a value here. If a PredicateNotSatisfied
        is used in the predicate, pass ``False``.
    :param progressbar: if True, show an automatic progress bar while waiting.
    :param caption: message to show in progress bar, and in TimeoutException (if ``message``
        not given).
    :param throw: if True, throw an exception if ``timeout`` expires.
        if ``pred`` not given, this is always False.
    :param allow_interruption: if True, the user can end the wait prematurely by hitting ESC.
    :param log_interval: interval for printing thrown ``PredicateNotSatisfied``s to the log.
        Set to ``None`` to disable this logging. If the predicate returns ``False`` instead
        of throwing this argument will be ignored. Defaults to 10 minutes.
    :param log_level: the log level for printing the thrown ``PredicateNotSatisfied`` with
        ``log_interval``. Defaults to ``logging.DEBUG``.
    """
    for ret in iter_wait(*args, **kwargs):
        pass
    return ret


def wait_progress(*args, **kwargs):
    for _ in iter_wait_progress(*args, **kwargs):
        pass


def iter_wait_progress(state_getter, advance_timeout, total_timeout=float("inf"), state_threshold=0, sleep=0.5, throw=True,
                       allow_regression=True, advancer_name=None, progressbar=True):

    ADVANCE_TIMEOUT_MESSAGE = "did not advance for {duration: .1f} seconds"
    TOTAL_TIMEOUT_MESSAGE = "advanced but failed to finish in {duration: .1f} seconds"

    state = state_getter()  # state_getter should return a number, represent current state

    progress = Bunch(state=state, finished=False, changed=False)
    progress.total_timer = Timer(expiration=total_timeout)
    progress.advance_timer = Timer(expiration=advance_timeout)

    def did_advance():
        current_state = state_getter()
        progress.advanced = progress.state > current_state
        progress.changed = progress.state != current_state
        if progress.advanced or allow_regression:
            progress.state = current_state
        return progress.advanced

    # We want to make sure each iteration sleeps at least once,
    # since the internal 'wait' could return immediately without sleeping at all,
    # and if the external while loop isn't done we could be iterating too much
    min_sleep = None

    while progress.state > state_threshold:
        progress.timeout, message = min(
            (progress.total_timer.remain, TOTAL_TIMEOUT_MESSAGE),
            (progress.advance_timer.remain, ADVANCE_TIMEOUT_MESSAGE))
        if advancer_name:
            message = advancer_name + ' ' + message

        if min_sleep:
            wait(min_sleep.remain)
        min_sleep = Timer(expiration=sleep)

        result = wait(progress.timeout, pred=did_advance, sleep=sleep, message=message, throw=throw, progressbar=progressbar)
        if not result:  # if wait times out without throwing
            return

        if progress.total_timer.expired:
            raise TimeoutException(message, duration=progress.total_timer.duration)

        progress.advance_timer.reset()
        yield progress

    progress.finished = True
    yield progress  # indicate success


from .timing import Timer  # noqa; avoid import cycle
