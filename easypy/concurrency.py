# encoding: utf-8

from concurrent.futures import ThreadPoolExecutor, CancelledError, as_completed, wait as futures_wait
from concurrent.futures import TimeoutError as FutureTimeoutError
import atexit
import sys
import threading
from easypy.threadtree import format_thread_stack
from itertools import chain
from functools import partial, wraps
from contextlib import contextmanager, ExitStack
from traceback import format_tb, extract_stack
import inspect
import os
from collections import defaultdict, Counter
import logging
import signal
from easypy.exceptions import TException, PException
from easypy.timing import Timer
from easypy.humanize import IndentableTextBuffer, time_duration
from easypy.decorations import parametrizeable_decorator
from easypy.misc import Hex
from easypy.units import MINUTE, HOUR
from threading import Event
from threading import RLock
import time

from importlib import import_module
this_module = import_module(__name__)


_logger = logging.getLogger(__name__)
_threads_logger = logging.getLogger('threads')

MAX_THREAD_POOL_SIZE = 50

_disabled = False


def disable():
    global _disabled
    _disabled = True
    logging.info("Concurrency disabled")


def enable():
    global _disabled
    _disabled = False
    logging.info("Concurrency enabled")


class ThreadTimeoutException(TException):
    template = "Thread timeout during execution {func}"


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


class MultiException(PException):

    template = "Exceptions raised from concurrent invocation ({0.common_type.__qualname__} x{0.count}/{0.invocations_count})"

    def __init__(self, exceptions, futures):
        # we want to keep futures in parallel with exceptions,
        # so some exceptions could be None
        assert len(futures) == len(exceptions)
        self.actual = list(filter(None, exceptions))
        self.count = len(self.actual)
        self.invocations_count = len(futures)
        self.common_type = concestor(*map(type, self.actual))
        self.one = self.actual[0] if self.actual else None
        self.futures = futures
        self.exceptions = exceptions
        self.complete = self.count == self.invocations_count
        if self.complete and hasattr(self.common_type, 'exit_with_code'):
            self.exit_with_code = self.common_type.exit_with_code
        super().__init__(self.template, self)

    def __repr__(self):
        return "{0.__class__.__name__}(<{0.common_type.__qualname__} x{0.count}/{0.invocations_count}>)".format(self)

    def __str__(self):
        return self.render(traceback=False, color=False)

    def walk(self, skip_multi_exceptions=True):
        if not skip_multi_exceptions:
            yield self
        for exc in self.actual:
            if isinstance(exc, MultiException):
                yield from exc.walk(skip_multi_exceptions=skip_multi_exceptions)
            else:
                yield exc

    def render(self, *, width=80, **kw):
        buff = self._get_buffer(**kw)
        return "\n"+buff.render(width=width)

    def _get_buffer(self, **kw):
        color = kw.get("color", True)
        buff = IndentableTextBuffer("{0.__class__.__qualname__}", self)
        if self.message:
            buff.write(("WHITE<<%s>>" % self.message) if color else self.message)

        traceback_fmt = "DARK_GRAY@{{{}}}@" if color else "{}"
        for exc in self.actual:
            with buff.indent("{.__class__.__qualname__}", exc):
                if isinstance(exc, self.__class__):
                    buff.extend(exc._get_buffer(**kw))
                elif hasattr(exc, "render"):
                    buff.write(exc.render(**kw))
                else:
                    if hasattr(exc, "context"):
                        context = "(%s)" % ", ".join("%s=%s" % p for p in sorted(exc.context.items()))
                    else:
                        context = ""
                    buff.write("{}: {}", exc, context)
                if hasattr(exc, "__traceback__"):
                    for line in format_tb(exc.__traceback__):
                        buff.write(traceback_fmt, line.rstrip())
        return buff


class Futures(list):

    def done(self):
        return all(f.done() for f in self)

    def cancelled(self):
        return all(f.cancelled() for f in self)

    def running(self):
        return all(f.running() for f in self)

    def wait(self, timeout=None):
        return futures_wait(self, timeout=timeout)

    def result(self, timeout=None):
        me = self.exception(timeout=timeout)
        if me:
            raise me
        return [f.result() for f in self]

    def exception(self, timeout=None):
        exceptions = [f.exception(timeout=timeout) for f in self]
        if any(exceptions):
            return MultiException(exceptions=exceptions, futures=self)

    def cancel(self):
        return all(f.cancel() for f in self)

    def as_completed(self, timeout=None):
        return as_completed(self, timeout=timeout)

    @classmethod
    @contextmanager
    def executor(cls, workers=MAX_THREAD_POOL_SIZE, ctx={}):
        futures = cls()
        with ThreadPoolExecutor(workers) as executor:
            def submit(func, *args, **kwargs):
                future = executor.submit(_run_with_exception_logging, func, args, kwargs, ctx)
                future.ctx = ctx
                future.funcname = _get_func_name(func)
                futures.append(future)
                return future
            futures.submit = submit
            futures.shutdown = executor.shutdown
            yield futures
        futures.result()  # bubble up any exceptions

    def dump_stacks(self, futures=None, verbose=False):
        futures = futures or self
        frames = sys._current_frames()
        for i, future in enumerate(futures, 1):
            try:
                frame = frames[future.ctx['thread_ident']]
            except KeyError:
                frame = None  # this might happen in race-conditions with a new thread starting
            if not verbose or not frame:
                if frame:
                    location = " - %s:%s, in %s(..)" % tuple(extract_stack(frame)[-1][:3])
                else:
                    location = "..."
                _logger.info("%3s - %s (DARK_YELLOW<<%s>>)%s",
                             i, future.funcname, _get_context(future), location)
                continue

            with _logger.indented("%3s - %s (%s)", i, future.funcname, _get_context(future), footer=False):
                lines = format_thread_stack(frame, this_module).splitlines()
                for line in lines:
                    _logger.info(line.strip())

    def logged_wait(self, timeout=None, initial_log_interval=None):
        log_interval = initial_log_interval or 2*MINUTE
        global_timer = Timer(expiration=timeout)
        iteration = 0

        while not global_timer.expired:
            completed, pending = self.wait(log_interval)
            if not pending:
                break

            iteration += 1
            if iteration % 5 == 0:
                log_interval *= 5
            with _logger.indented("(Waiting for %s on %s/%s tasks...)",
                                  time_duration(global_timer.elapsed),
                                  len(pending), sum(map(len, (completed, pending))),
                                  level=logging.WARNING, footer=False):
                self.dump_stacks(pending, verbose=global_timer.elapsed >= HOUR)


def _run_with_exception_logging(func, args, kwargs, ctx):
    with _logger.context(**ctx):
        ctx['thread_ident'] = Hex(threading.current_thread().ident)
        try:
            return func(*args, **kwargs)
        except StopIteration:
            # no need to log this
            raise
        except Exception:
            _logger.silent_exception("Exception in thread running %s (traceback in debug logs)", func)
            raise


def _to_args_list(params):
    return [args if isinstance(args, tuple) else (args,) for args in params]


def _get_func_name(func):
    kw = {}
    while isinstance(func, partial):
        if func.keywords:
            kw.update(func.keywords)
        func = func.func
    funcname = func.__qualname__
    if kw:
        funcname += "(%s)" % ", ".join("%s=%r" % p for p in sorted(kw.items()))
    return funcname


def _to_log_contexts(params, log_contexts):
    if not log_contexts:
        log_contexts = (dict(context=str(p) if len(p) > 1 else str(p[0])) for p in params)
    else:
        log_contexts = (p if isinstance(p, dict) else dict(context=str(p))
                        for p in log_contexts)
    return log_contexts


def _get_context(future):
    ctx = dict(future.ctx)
    context = "%X;" % ctx.pop("thread_ident", 0)
    context += ctx.pop("context", "")
    context += ";".join("%s=%s" % p for p in sorted(ctx.items()))
    return context


@contextmanager
def async(func, params=None, workers=None, log_contexts=None, final_timeout=2.0, **kw):
    if params is None:
        params = [()]
    if not isinstance(params, list):
        params = [params]
    params = _to_args_list(params)
    log_contexts = _to_log_contexts(params, log_contexts)

    workers = workers or min(MAX_THREAD_POOL_SIZE, len(params))
    executor = ThreadPoolExecutor(workers) if workers else None

    funcname = _get_func_name(func)

    futures = Futures()
    for args, ctx in zip(params, log_contexts):
        future = executor.submit(_run_with_exception_logging, func, args, {}, ctx)
        future.ctx = ctx
        future.funcname = funcname
        futures.append(future)

    def kill(wait=False):
        nonlocal killed
        futures.cancel()
        if executor:
            executor.shutdown(wait=wait)
        killed = True

    killed = False
    futures.kill = kill

    try:
        yield futures
    except:
        _logger.debug("shutting down ThreadPoolExecutor due to exception")
        kill(wait=False)
        raise
    else:
        if executor:
            executor.shutdown(wait=not killed)
        if not killed:
            # force exceptions to bubble up
            try:
                futures.result(timeout=final_timeout)
            except CancelledError:
                pass
    finally:
        # break the cycle so that the GC doesn't clean up the executor under a lock (https://bugs.python.org/issue21009)
        futures.kill = None
        futures = None


def concurrent_find(func, params, **kw):
    timeout = kw.pop("concurrent_timeout", None)
    with async(func, list(params), **kw) as futures:
        future = None
        try:
            for future in futures.as_completed(timeout=timeout):
                if not future.exception() and future.result():
                    futures.kill()
                    return future.result()
            else:
                if future:
                    future.result()
        except FutureTimeoutError as exc:
            if not timeout:
                # ??
                raise
            futures.kill()
            _logger.warning("Concurrent future timed out (%s)", exc)


def nonconcurrent_map(func, params, log_contexts=None, **kw):
    futures, results, exceptions = [], [], []
    log_contexts = _to_log_contexts(params, log_contexts)
    for args, ctx in zip(_to_args_list(params), log_contexts):
        f = partial(_run_with_exception_logging, func, args, kw, ctx)
        futures.append(f)
        try:
            results.append(f())
        except Exception as exc:
            exceptions.append(exc)
        else:
            exceptions.append(None)
    if any(exceptions):
        raise MultiException(exceptions=exceptions, futures=futures)
    del futures[:]
    return results


def concurrent_map(func, params, workers=None, log_contexts=None, initial_log_interval=None, **kw):
    if _disabled or len(params) == 1:
        return nonconcurrent_map(func, params, log_contexts, **kw)

    with async(func, list(params), workers, log_contexts, **kw) as futures:
        futures.logged_wait(initial_log_interval=initial_log_interval)
        return futures.result()


class MultiObject(object):

    def __init__(self, items=None, log_ctx=None, workers=None):
        self._items = list(items) if items else []
        self._workers = workers
        cstr = concestor(*map(type, self))
        if hasattr(cstr, '_multiobject_log_ctx'):
            # override the given log_ctx if the new items have it
            # some objects (Plumbum Cmd) are expensive to just get the attribute, so we require it
            # on the base class
            self._log_ctx = [item._multiobject_log_ctx for item in self._items]
        elif callable(log_ctx):
            self._log_ctx = list(map(log_ctx, self._items))
        elif log_ctx:
            self._log_ctx = list(log_ctx)
        elif issubclass(cstr, str):
            self._log_ctx = [dict(context="%s" % item) for item in self._items]
        else:
            self._log_ctx = [dict(context="%s<M%03d>" % (cstr.__name__, i)) for i, item in enumerate(self._items)]

        if self._workers is None and hasattr(cstr, '_multiobject_workers'):
            _workers = cstr._multiobject_workers
            if _workers == -1:
                self._workers = len(self._items) or None
            else:
                self._workers = _workers

    @property
    def L(self):
        return list(self)

    @property
    def C(self):
        from .collections import ListCollection
        return ListCollection(self)

    def __getattr__(self, attr):
        if attr.startswith("_"):
            raise AttributeError()
        get = lambda obj: getattr(obj, attr)
        ret = concurrent_map(get, self, log_contexts=self._log_ctx, workers=self._workers)
        return self._new(ret)

    def __call__(self, *args, **kwargs):
        if not self:
            return self._new(self)
        for obj in self:
            if not callable(obj):
                raise Exception("%s is not callable" % obj)

        def do_it(obj):
            return obj(*args, **kwargs)

        if all(hasattr(obj, "__qualname__") for obj in self):
            do_it = wraps(obj)(do_it)
        else:
            common_typ = concestor(*map(type, self))
            do_it.__qualname__ = common_typ.__qualname__
        initial_log_interval = kwargs.pop("initial_log_interval", None)
        ret = concurrent_map(
            do_it, self,
            log_contexts=self._log_ctx,
            workers=self._workers,
            initial_log_interval=initial_log_interval)
        return self._new(ret)

    def __dir__(self):
        return sorted(set.intersection(*(set(dir(obj)) for obj in self)))
    trait_names = __dir__

    def __iter__(self):
        return iter(self._items)

    def __getitem__(self, key):
        ret = self._items[key]
        if isinstance(key, slice):
            return self._new(ret, self._log_ctx[key])
        else:
            return ret

    # === Mutation =====
    # TODO: ensure the relationship between items/workers/logctx
    def __delitem__(self, key):
        del self._log_ctx[key]
        del self._items[key]

    def __len__(self):
        return len(self._items)

    def __add__(self, other):
        return self.__class__(chain(self, other))

    def __iadd__(self, other):
        self._log_ctx.extend(other._log_ctx)
        return self._items.extend(other)

    def sort(self, *args, **kwargs):
        order = {id(obj): log_ctx for (obj, log_ctx) in zip(self._items, self._log_ctx)}
        ret = self._items.sort(*args, **kwargs)
        self._log_ctx[:] = [order[id(obj)] for obj in self._items]
        return ret

    def append(self, item, *, log_ctx=None):
        self._log_ctx.append(log_ctx)
        return self._items.append(item)

    def insert(self, pos, item, *, log_ctx=None):
        self._log_ctx.insert(pos, log_ctx)
        return self._items.insert(pos, item)

    def pop(self, *args):
        self._log_ctx.pop(*args)
        return self._items.pop(*args)

    # ================

    def __repr__(self):
        common_typ = concestor(*map(type, self))
        if common_typ:
            return "<MultiObject '%s' (x%s/%s)>" % (common_typ.__name__, len(self), self._workers)
        else:
            return "<MultiObject (Empty)>"

    def _new(self, items=None, ctxs=None, workers=None):
        return self.__class__(
            self._items if items is None else items,
            self._log_ctx if ctxs is None else ctxs,
            self._workers if workers is None else workers)

    def with_workers(self, workers):
        "Return a new MultiObject based on current items with the specified number of workers"
        return self._new(workers=workers)

    def call(self, func, **kw):
        "Concurrently call a function on each of the object contained by this MultiObject (as first param)"
        initial_log_interval = kw.pop("initial_log_interval", None)
        if kw:
            func = wraps(func)(partial(func, **kw))
        return self._new(concurrent_map(
            func, self,
            log_contexts=self._log_ctx,
            workers=self._workers,
            initial_log_interval=initial_log_interval))
    each = call

    def filter(self, pred):
        if not pred:
            pred = bool
        filtering = self.call(pred)
        filtered = [t for (*t, passed) in zip(self, self._log_ctx, filtering) if passed]
        return self._new(*(zip(*filtered) if filtered else ((),())))

    def chain(self):
        "Chain the iterables contained by this MultiObject"
        return self.__class__(chain(*self))

    def zip(self):
        "Concurrently iterate through the iterables contained by this MultiObject"
        iters = list(map(iter, self))
        while True:
            try:
                ret = concurrent_map(next, iters, log_contexts=self._log_ctx, workers=self._workers)
            except MultiException as me:
                if me.common_type == StopIteration and me.complete:
                    break
                raise
            else:
                yield self._new(ret)

    def concurrent_find(self, func=lambda f: f(), **kw):
        return concurrent_find(func, self, log_contexts=self._log_ctx, workers=self._workers, **kw)

    def __enter__(self):
        return self.call(lambda obj: obj.__enter__())

    def __exit__(self, *args):
        self.call(lambda obj: obj.__exit__(*args))


def concestor(*cls_list):
    "Closest common ancestor class"
    mros = [list(inspect.getmro(cls)) for cls in cls_list]
    track = defaultdict(int)
    while mros:
        for mro in mros:
            cur = mro.pop(0)
            track[cur] += 1
            if track[cur] == len(cls_list):
                return cur
            if len(mro) == 0:
                mros.remove(mro)
    return object  # the base-class that rules the all


LAST_ERROR = None
REGISTERD_SIGNAL = None


class Error(TException):
    pass


class NotMainThread(Error):
    template = "Binding must be invoked from main thread"


class SignalAlreadyBound(Error):
    template = "Signal already bound to another signal handler(s)"


class LastErrorEmpty(Error):
    template = "Signal caught, but no error to raise"


class NotInitialized(Error):
    template = "Signal type not initialized, must use bind_to_subthread_exceptions in the main thread"


class TerminationSignal(TException):
    template = "Process got a termination signal: {_signal}"


def initialize_exception_listener():
    global REGISTERD_SIGNAL
    if REGISTERD_SIGNAL:
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
        REGISTERD_SIGNAL = custom_signal
    else:
        raise SignalAlreadyBound(signal=custom_signal)


@contextmanager
def raise_in_main_thread(exception_type=Exception):
    from plumbum import local
    pid = os.getpid()
    if not REGISTERD_SIGNAL:
        raise NotInitialized()
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

        # sometimes the signal isn't caught by the main-thread, so we should try a few times (WEKAPP-14543)
        def do_signal(raised_exc):
            global LAST_ERROR
            if LAST_ERROR is not raised_exc:
                _logger.debug("MainThread took the exception - we're done here")
                raiser.stop()
                return

            _logger.info("Raising %s in main thread", type(LAST_ERROR))
            local.cmd.kill("-%d" % REGISTERD_SIGNAL, pid)

        raiser = concurrent(do_signal, raised_exc=exc, loop=True, sleep=30, daemon=True, throw=False)
        raiser.start()

        raise


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

    def __init__(self, timeout, alert_interval=None):
        self.fuse = Event()  # use this to cancel the timebomb
        self.timeout = timeout
        self.alert_interval = alert_interval

    def __enter__(self):
        return self.start()

    def __exit__(self, *args):
        self.cancel()

    def start(self):
        self.t = concurrent(self.wait_and_kill, daemon=True, threadname="Timebomb(%s)" % self.timeout)
        self.t.start()
        return self

    def cancel(self):
        self.fuse.set()

    @raise_in_main_thread()
    def wait_and_kill(self):
        timer = Timer(expiration=self.timeout)
        _logger.info("Timebomb set - this process will YELLOW<<self-destruct>> in RED<<%r>>...", timer.remain)
        while not timer.expired:
            if self.alert_interval:
                _logger.info("Time Elapsed: MAGENTA<<%r>>", timer.elapsed)
            log_level = logging.WARNING if timer.remain < 5*MINUTE else logging.DEBUG
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


class concurrent(object):

    def __init__(self, func, *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.throw = kwargs.pop('throw', True)
        self.daemon = kwargs.pop('daemon', True)
        self.threadname = kwargs.pop('threadname', None)
        self.stopper = kwargs.pop('stopper', Event())
        self.sleep = kwargs.pop('sleep', 1)
        self.loop = kwargs.pop('loop', False)
        self.timer = None

        rimt = kwargs.pop("raise_in_main_thread", False)
        if rimt:
            exc_type = Exception if rimt is True else rimt
            self.func = raise_in_main_thread(exc_type)(self.func)

    def _logged_func(self):
        self.timer = Timer()
        try:
            while True:
                self.result = self.func(*self.args, **self.kwargs)
                if not self.loop:
                    return
                if self.wait(self.sleep):
                    return
        except Exception as exc:
            _logger.silent_exception("Exception in thread running %s (traceback can be found in debug-level logs)", self.func)
            self.exc = exc
        finally:
            self.timer.stop()
            self.stop()

    def stop(self):
        self.stopper.set()

    def wait(self, timeout=None):
        return self.stopper.wait(timeout)

    @contextmanager
    def paused(self):
        self.stop()
        yield
        self.start()

    @contextmanager
    def _running(self):
        self.thread = threading.Thread(target=self._logged_func, name=self.threadname)
        self.thread.daemon = self.daemon
        self.exc = None
        self.stopper.clear()

        if _disabled:
            self._logged_func()
            yield self
            return

        self.thread.start()
        try:
            yield self
        finally:
            self.stop()  # if we loop, stop it
        self.thread.join()
        if self.throw and self.exc:
            raise self.exc

    def __enter__(self):
        self._ctx = self._running()
        return self._ctx.__enter__()

    def __exit__(self, *args):
        return self._ctx.__exit__(*args)

    def __iter__(self):
        with self:
            self.iterations = 0
            while not self.wait(self.sleep):
                yield self
                self.iterations += 1

    start = __enter__

    def join(self):
        self.__exit__(None, None, None)

    __del__ = join


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


def _get_my_ident():
    return Hex(threading.current_thread().ident)


class SoftLock(object):

    def __init__(self):
        self.lock = RLock()
        self.cond = threading.Condition(self.lock)
        self.owners = Counter()

    def __repr__(self):
        owners = ", ".join(map(str, sorted(self.owners.keys())))
        return "<SoftLock-{:X}, owned by [{}]>".format(id(self.lock), owners)

    @property
    def owner_count(self):
        return sum(self.owners.values())

    def __call__(self):
        return self

    def __enter__(self):
        try:
            while not self.cond.acquire(timeout=15):
                _logger.debug("waiting on %s", self)
            self.owners[_get_my_ident()] += 1
            _logger.debug("aquired %s non-exclusively", self)
            return self
        finally:
            self.cond.release()

    def __exit__(self, *args):
        try:
            while not self.cond.acquire(timeout=15):
                _logger.debug("waiting on %s", self)
            my_ident = _get_my_ident()
            self.owners[my_ident] -= 1
            if not self.owners[my_ident]:
                self.owners.pop(my_ident)  # don't inflate the soft lock keys with threads that does not own it
            self.cond.notify()
            _logger.debug("released bob exclusive lock on %s", self)
        finally:
            self.cond.release()

    @contextmanager
    def exclusive(self, need_to_wait_message=None):
        with self.cond:
            # wait until this thread is the sole owner of this lock
            while not self.cond.wait_for(lambda: self.owner_count == self.owners[_get_my_ident()], timeout=15):
                _check_exiting()
                if need_to_wait_message:
                    _logger.info(need_to_wait_message)
                    need_to_wait_message = None  # only print it once
                _logger.debug("waiting for exclusivity on %s", self)
            my_ident = _get_my_ident()
            self.owners[my_ident] += 1
            _logger.debug("%s - acquired exclusively", self)
            try:
                yield
            finally:
                self.owners[my_ident] -= 1
                if not self.owners[my_ident]:
                    self.owners.pop(my_ident)  # don't inflate the soft lock keys with threads that does not own it
                self.cond.notify()
                _logger.debug('releasing exclusive lock on %s', self)


class TagAlongThread(object):

    def __init__(self, func, name, minimal_sleep=0):
        self._func = func
        self.minimal_sleep = minimal_sleep

        self._iteration_trigger = Event()

        self._iterating = Event()
        self._not_iterating = Event()
        self._not_iterating.set()

        self._last_exception = None
        self._last_result = None

        self._thread = threading.Thread(target=self._loop, daemon=True, name=name)
        self._thread.start()

    def _loop(self):
        while True:
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

    def __call__(self):
        # We can't use an iteration that's already started - maybe it's already at a too advanced stage?
        if self._iterating.is_set():
            self._not_iterating.wait()

        self._iteration_trigger.set()  # Signal that we want an iteration

        self._iterating.wait()  # Wait until an iteration starts
        self._not_iterating.wait()  # Wait until it finishes

        # To avoid races, copy last exception and result to local variables
        last_exception, last_result = self._last_exception, self._last_result
        if last_exception:
            raise last_exception
        else:
            return last_result


def throttled(func, ms):
    last_run = 0
    lock = RLock()

    @wraps(func)
    def throttled_func():
        nonlocal last_run
        ts = time.time() * 1000
        run = False
        with lock:
            if ts - last_run > ms:
                last_run = ts
                run = True
        if run:
            return func()
    return throttled_func


class synchronized_on_first_call():
    "Decorator, that make a function synchronized but only on its first invocation"

    def __init__(self, func):
        self.lock = RLock()
        self.func = func
        self.initialized = False

    def __call__(self, *args, **kwargs):
        with ExitStack() as stack:
            with self.lock:
                if not self.initialized:
                    stack.enter_context(self.lock)
            ret = self.func(*args, **kwargs)
            if not self.initialized:
                self.initialized = True
            return ret
