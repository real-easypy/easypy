# encoding: utf-8

"""
This module helps you run things concurrently
"""

from concurrent.futures import ThreadPoolExecutor, CancelledError, as_completed, Future, wait as futures_wait
from concurrent.futures import TimeoutError as FutureTimeoutError

from collections import defaultdict
from contextlib import contextmanager, ExitStack
from functools import partial, wraps
from importlib import import_module
from itertools import chain
from traceback import format_tb
import inspect
import logging
import threading
import time
from collections import namedtuple
from datetime import datetime

from easypy.exceptions import PException
from easypy.gevent import is_module_patched, non_gevent_sleep, defer_to_thread
from easypy.humanize import IndentableTextBuffer, time_duration, compact
from easypy.humanize import format_thread_stack
from easypy.threadtree import iter_thread_frames
from easypy.timing import Timer
from easypy.units import MINUTE, HOUR
from easypy.colors import colorize_by_patterns
from easypy.sync import SynchronizationCoordinator, ProcessExiting, THREADING_MODULE_PATHS, raise_in_main_thread


this_module = import_module(__name__)


MAX_THREAD_POOL_SIZE = 50


try:
    from traceback import _extract_stack_iter
except ImportError:
    from traceback import walk_stack

    def _extract_stack_iter(frame):
        for f, lineno in walk_stack(frame):
            co = f.f_code
            filename = co.co_filename
            name = co.co_name
            yield filename, lineno, name


_logger = logging.getLogger(__name__)

_disabled = False


def disable():
    global _disabled
    _disabled = True
    logging.info("Concurrency disabled")


def enable():
    global _disabled
    _disabled = False
    logging.info("Concurrency enabled")


def _find_interesting_frame(f):
    default = next(_extract_stack_iter(f))
    non_threading = (
        p for p in _extract_stack_iter(f)
        if all(not p[0].startswith(pth) for pth in THREADING_MODULE_PATHS))
    return next(non_threading, default)


# This metaclass helps generate MultiException subtypes so that it's easier
# to catch a MultiException with a specific common type
class MultiExceptionMeta(type):
    _SUBTYPES = {}
    _SUBTYPES_LOCK = threading.RLock()

    def __getitem__(cls, exception_type):
        if exception_type is BaseException:
            return MultiException

        assert isinstance(exception_type, type), "Must use an Exception type"
        assert issubclass(exception_type, BaseException), "Must inherit for BaseException"

        try:
            return cls._SUBTYPES[exception_type]
        except KeyError:
            with cls._SUBTYPES_LOCK:
                if exception_type in cls._SUBTYPES:
                    return cls._SUBTYPES[exception_type]
                bases = tuple(cls[base] for base in exception_type.__bases__ if base and issubclass(base, BaseException))
                subtype = type("MultiException[%s]" % exception_type.__qualname__, bases, dict(COMMON_TYPE=exception_type))
                cls._SUBTYPES[exception_type] = subtype
                return subtype

    __iter__ = None

    def __call__(cls, exceptions, futures):
        common_type = concestor(*map(type, filter(None, exceptions)))
        subtype = cls[common_type]
        return type.__call__(subtype, exceptions, futures)


PickledFuture = namedtuple("PickledFuture", "ctx, funcname")


class MultiException(PException, metaclass=MultiExceptionMeta):

    template = "{0.common_type.__qualname__} raised from concurrent invocation (x{0.count}/{0.invocations_count})"

    def __reduce__(self):
        return (MultiException, (self.exceptions, [PickledFuture(ctx=f.ctx, funcname=f.funcname) for f in self.futures]))

    def __init__(self, exceptions, futures):
        # we want to keep futures in parallel with exceptions,
        # so some exceptions could be None
        assert len(futures) == len(exceptions)
        self.actual = list(filter(None, exceptions))
        self.count = len(self.actual)
        self.invocations_count = len(futures)
        self.common_type = self.COMMON_TYPE
        self.one = self.actual[0] if self.actual else None
        self.futures = futures
        self.exceptions = exceptions
        self.complete = self.count == self.invocations_count
        if self.complete and hasattr(self.common_type, 'exit_with_code'):
            self.exit_with_code = self.common_type.exit_with_code
        super().__init__(self.template, self)

    def __repr__(self):
        return "{0.__class__.__name__}(x{0.count}/{0.invocations_count})".format(self)

    def __str__(self):
        return self.render(color=False)

    def walk(self, skip_multi_exceptions=True):
        if not skip_multi_exceptions:
            yield self
        for exc in self.actual:
            if isinstance(exc, MultiException):
                yield from exc.walk(skip_multi_exceptions=skip_multi_exceptions)
            else:
                yield exc

    def render(self, *, width=80, color=True, **kw):
        buff = self._get_buffer(color=color, **kw)
        text = buff.render(width=width, edges=not color)
        return colorize_by_patterns("\n" + text)

    def _get_buffer(self, **kw):
        if kw.get("color", True):
            normalize_color = lambda x: x
        else:
            normalize_color = partial(colorize_by_patterns, no_color=True)

        def _format_context(context):
            if not isinstance(context, dict):
                return repr(context)  # when it comes from rpyc
            context = context.copy()
            context.pop("indentation", None)
            breadcrumbs = ";".join(context.pop('context', []))
            return ", ".join(filter(None, chain((breadcrumbs,), ("%s=%s" % p for p in sorted(context.items())))))

        buff = IndentableTextBuffer("{0.__class__.__qualname__}", self)
        if self.message:
            buff.write(normalize_color("WHITE<<%s>>" % self.message))

        traceback_fmt = normalize_color("DARK_GRAY<<{}>>")

        # workaround incompatibilty with rpyc, which serializes .actual into an str
        # instead of a list of exceptions. This makes the string flatten into a long
        # and incomprehensible text buffer.
        if hasattr(self, "_remote_tb"):
            with buff.indent("Remote Traceback:"):
                buff.write(self._remote_tb)
            return buff

        def add_details(exc):
            if kw.get("timestamp", True) and getattr(exc, "timestamp", None):
                ts = datetime.fromtimestamp(exc.timestamp).isoformat()
                buff.write(normalize_color("MAGENTA<<Timestamp: %s>>" % ts))
            if kw.get("context", True) and getattr(exc, "context", None):
                buff.write("Context: %s" % _format_context(exc.context))

        add_details(self)

        for exc in self.actual:
            with buff.indent("{.__class__.__qualname__}", exc):
                if isinstance(exc, MultiException):
                    buff.extend(exc._get_buffer(**kw))
                elif callable(getattr(exc, "render", None)):
                    buff.write(exc.render(**kw))
                else:
                    buff.write("{}", exc)
                    add_details(exc)
                if hasattr(exc, "__traceback__"):
                    show_traceback = getattr(exc, 'traceback', None)
                    if show_traceback is not False:
                        buff.write("Traceback:")
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
            if isinstance(me, MultiException[ProcessExiting]):
                # we want these aborted MultiObject threads to consolidate this exception
                raise ProcessExiting()
            raise me
        return [f.result() for f in self]

    def exception(self, timeout=None):
        exceptions = [f.exception(timeout=timeout) for f in self]
        if any(exceptions):
            return MultiException(exceptions=exceptions, futures=self)

    def cancel(self):
        cancelled = [f.cancel() for f in self]  # list-comp, to ensure we call cancel on all futures
        return all(cancelled)

    def as_completed(self, timeout=None):
        return as_completed(self, timeout=timeout)

    @classmethod
    @contextmanager
    def executor(cls, workers=MAX_THREAD_POOL_SIZE, ctx={}):
        futures = cls()
        with ThreadPoolExecutor(workers) as executor:
            def submit(func, *args, log_ctx={}, **kwargs):
                _ctx = dict(ctx, **log_ctx)
                future = executor.submit(_run_with_exception_logging, func, args, kwargs, _ctx)
                future.ctx = _ctx
                future.funcname = _get_func_name(func)
                futures.append(future)
                return future
            futures.submit = submit
            futures.shutdown = executor.shutdown
            yield futures
        futures.result()  # bubble up any exceptions

    def dump_stacks(self, futures=None, verbose=False):
        futures = futures or self
        frames = dict(iter_thread_frames())

        for i, future in enumerate(futures, 1):
            try:
                frame = frames[future.ctx['thread_ident']]
            except KeyError:
                frame = None  # this might happen in race-conditions with a new thread starting
            if not verbose or not frame:
                if frame:
                    frame_line = _find_interesting_frame(frame)[:3]
                    location = " - %s:%s, in %s(..)" % tuple(frame_line)
                else:
                    location = "..."
                _logger.info("%3s - %s (DARK_YELLOW<<%s>>)%s",
                             i, future.funcname, _get_context(future), location)
                continue

            with _logger.indented("%3s - %s (%s)", i, future.funcname, _get_context(future), footer=False):
                lines = format_thread_stack(frame, skip_modules=[this_module]).splitlines()
                for line in lines:
                    _logger.info(line.strip())

    def logged_wait(self, timeout=None, initial_log_interval=None):
        log_interval = initial_log_interval or 2 * MINUTE
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
    thread = threading.current_thread()
    ctx.update(threadname=thread.name, thread_ident=thread.ident)
    with _logger.context(**ctx):
        try:
            return func(*args, **kwargs)
        except StopIteration:
            # no need to log this
            raise
        except ProcessExiting as exc:
            _logger.debug(exc)
            raise
        except Exception as exc:
            _logger.silent_exception(
                "Exception (%s) in thread running %s (traceback in debug logs)",
                exc.__class__.__qualname__, func)
            try:
                exc.timestamp = time.time()
            except:  # noqa - sometimes exception objects are immutable
                pass
            raise


def _to_args_list(params):
    # We use type(args) == tuple because isinstance will return True for namedtuple
    return [args if type(args) == tuple else (args,) for args in params]


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
    def compacted(s):
        return compact(str(s).split("\n", 1)[0], 20, "....", 5).strip()
    ctx = dict(future.ctx)
    context = []
    threadname = ctx.pop("threadname", None)
    thread_ident = ctx.pop("thread_ident", None)
    context.append(threadname or thread_ident)
    context.append(ctx.pop("context", None))
    context.extend("%s=%s" % (k, compacted(v)) for k, v in sorted(ctx.items()))
    return ";".join(filter(None, context))


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

    try:
        signature = inspect.signature(func)
    except ValueError:
        # In Python 3.5+, inspect.signature returns this for built-in types
        pass
    else:
        if '_sync' in signature.parameters and '_sync' not in kw:
            assert len(params) <= executor._max_workers, 'SynchronizationCoordinator with %s tasks but only %s workers' % (
                len(params), executor._max_workers)
            synchronization_coordinator = SynchronizationCoordinator(len(params))
            kw['_sync'] = synchronization_coordinator

            func = synchronization_coordinator._abandon_when_done(func)

    futures = Futures()
    for args, ctx in zip(params, log_contexts):
        future = executor.submit(_run_with_exception_logging, func, args, kw, ctx)
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
                    return future.result()
        except FutureTimeoutError as exc:
            if not timeout:
                # ??
                raise
            futures.kill()
            _logger.warning("Concurrent future timed out (%s)", exc)


def nonconcurrent_map(func, params, log_contexts=None, **kw):
    futures = Futures()
    log_contexts = _to_log_contexts(params, log_contexts)
    has_exceptions = False
    for args, ctx in zip(_to_args_list(params), log_contexts):
        future = Future()
        futures.append(future)
        try:
            result = _run_with_exception_logging(func, args, kw, ctx)
        except Exception as exc:
            has_exceptions = True
            future.set_exception(exc)
        else:
            future.set_result(result)

    if has_exceptions:
        exceptions = [f.exception() for f in futures]
        raise MultiException(exceptions=exceptions, futures=futures)

    results = [f.result() for f in futures]
    del futures[:]
    return results


def concurrent_map(func, params, workers=None, log_contexts=None, initial_log_interval=None, **kw):
    if _disabled or len(params) == 1:
        return nonconcurrent_map(func, params, log_contexts, **kw)

    with async(func, list(params), workers, log_contexts, **kw) as futures:
        futures.logged_wait(initial_log_interval=initial_log_interval)
        return futures.result()


# This metaclass helps generate MultiObject subtypes for specific object types

class MultiObjectMeta(type):
    _SUBTYPES = {}
    _SUBTYPES_LOCK = threading.RLock()

    def __getitem__(cls, typ):
        try:
            return cls._SUBTYPES[typ]
        except KeyError:
            with cls._SUBTYPES_LOCK:
                if typ in cls._SUBTYPES:
                    return cls._SUBTYPES[typ]
                bases = tuple(cls[base] for base in typ.__bases__ if base) or (MultiObject, )
                subtype = type("MultiObject[%s]" % typ.__qualname__, bases, dict(CONCESTOR=typ))
                cls._SUBTYPES[typ] = subtype
                return subtype

    __iter__ = None

    def __call__(cls, items=None, *args, **kwargs):
        items = tuple(items if items else [])
        common_type = concestor(*map(type, items))
        if not issubclass(common_type, cls.CONCESTOR):
            raise TypeError("%s is not a type of %s" % (common_type, cls.CONCESTOR))
        subtype = cls[common_type]
        return type.__call__(subtype, items, *args, **kwargs)


class MultiObject(object, metaclass=MultiObjectMeta):

    CONCESTOR = object

    def __init__(self, items=None, log_ctx=None, workers=None, initial_log_interval=None):
        self._items = tuple(items) if items else ()
        self._workers = workers
        self._initial_log_interval = initial_log_interval
        cstr = self.CONCESTOR
        if hasattr(cstr, '_multiobject_log_ctx'):
            # override the given log_ctx if the new items have it
            # some objects (Plumbum Cmd) are expensive to just get the attribute, so we require it
            # on the base class
            self._log_ctx = tuple(item._multiobject_log_ctx for item in self._items)
        elif callable(log_ctx):
            self._log_ctx = tuple(map(log_ctx, self._items))
        elif log_ctx:
            self._log_ctx = tuple(log_ctx)
        elif issubclass(cstr, str):
            self._log_ctx = tuple(dict(context="%s" % item) for item in self._items)
        else:
            self._log_ctx = tuple(dict(context="%s<M%03d>" % (cstr.__name__, i)) for i, item in enumerate(self._items))

        if self._workers is None and hasattr(cstr, '_multiobject_workers'):
            _workers = cstr._multiobject_workers
            if _workers == -1:
                self._workers = len(self._items) or None
            else:
                self._workers = _workers

    def __repr__(self):
        return "<%s (x%s/%s)>" % (self.__class__.__name__, len(self), self._workers)

    @property
    def L(self):
        return list(self._items)

    @property
    def T(self):
        return self._items

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

        def do_it(obj, **more_kwargs):
            more_kwargs.update(kwargs)
            return obj(*args, **more_kwargs)

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
        return sorted(set.intersection(*(set(dir(obj)) for obj in self)).union(super().__dir__()))

    trait_names = __dir__

    def __iter__(self):
        return iter(self._items)

    def __len__(self):
        return len(self._items)

    # ================
    def __getitem__(self, key):
        return self.call(lambda i: i[key])

    def _new(self, items=None, ctxs=None, workers=None, initial_log_interval=None):
        return MultiObject(
            self._items if items is None else items,
            self._log_ctx if ctxs is None else ctxs,
            self._workers if workers is None else workers,
            self._initial_log_interval if initial_log_interval is None else initial_log_interval)

    def with_workers(self, workers):
        "Return a new ``MultiObject`` based on current items with the specified number of workers"
        return self._new(workers=workers)

    def call(self, func, *args, **kw):
        "Concurrently call a function on each of the object contained by this ``MultiObject`` (as first param)"
        initial_log_interval = kw.pop("initial_log_interval", self._initial_log_interval)
        if kw:
            func = wraps(func)(partial(func, **kw))
        params = [((item,) + args) for item in self] if args else self
        return self._new(concurrent_map(
            func, params,
            log_contexts=self._log_ctx,
            workers=self._workers,
            initial_log_interval=initial_log_interval), initial_log_interval=initial_log_interval)

    each = call

    def filter(self, pred):
        if not pred:
            pred = bool
        filtering = self.call(pred)
        filtered = [t for (*t, passed) in zip(self, self._log_ctx, filtering) if passed]
        return self._new(*(zip(*filtered) if filtered else ((), ())))

    def chain(self):
        "Chain the iterables contained by this ``MultiObject``"
        return MultiObject(chain(*self))

    def zip_with(self, *collections):
        mo = self._new(zip(self, *collections))
        assert len(mo) == len(self), "All collection must have at least %s items" % len(self)
        return mo

    def enumerate(self, start=0):
        """
        Replaces this pattern, which loses the log contexts::

            MultiObject(enumerate(items)).call(lambda idx, item: ...)

        with this pattern, which retains log contexts::

            MultiObject(items).enumerate().call(lambda idx, item: ...)
        """
        return self._new(zip(range(start, start + len(self)), self))

    def zip(self):
        "Concurrently iterate through the iterables contained by this ``MultiObject``"
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


class concurrent(object):

    def __init__(self, func, *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.throw = kwargs.pop('throw', True)
        self.daemon = kwargs.pop('daemon', True)
        self.threadname = kwargs.pop('threadname', 'anon-%X' % id(self))
        self.stopper = kwargs.pop('stopper', threading.Event())
        self.sleep = kwargs.pop('sleep', 1)
        self.loop = kwargs.pop('loop', False)
        self.timer = None
        self.console_logging = kwargs.pop('console_logging', True)

        real_thread_no_greenlet = kwargs.pop('real_thread_no_greenlet', False)
        if is_module_patched("threading"):
            # in case of using apply_gevent_patch function - use this option in order to defer some jobs to real threads
            self.real_thread_no_greenlet = real_thread_no_greenlet
        else:
            # gevent isn't active, no need to do anything special
            self.real_thread_no_greenlet = False

        rimt = kwargs.pop("raise_in_main_thread", False)
        if rimt:
            exc_type = Exception if rimt is True else rimt
            self.func = raise_in_main_thread(exc_type)(self.func)

    def __repr__(self):
        flags = ""
        if self.daemon:
            flags += 'D'
        if self.loop:
            flags += 'L'
        if self.real_thread_no_greenlet:
            flags += 'T'
        return "<%s[%s] '%s'>" % (self.__class__.__name__, self.threadname, flags)

    def _logged_func(self):
        stack = ExitStack()
        self.exc = None
        self.timer = Timer()
        stack.callback(self.timer.stop)
        stack.callback(self.stop)
        try:
            if not self.console_logging:
                stack.enter_context(_logger.suppressed())
            _logger.debug("%s - starting", self)
            while True:
                self.result = self.func(*self.args, **self.kwargs)
                if not self.loop:
                    return
                if self.wait(self.sleep):
                    _logger.debug("%s - stopped", self)
                    return
        except ProcessExiting as exc:
            _logger.debug(exc)
            raise
        except Exception as exc:
            _logger.silent_exception("Exception in thread running %s (traceback can be found in debug-level logs)", self.func)
            self.exc = exc
            try:
                exc.timestamp = time.time()
            except Exception:
                pass
        finally:
            stack.close()

    def stop(self):
        _logger.debug("%s - stopping", self)
        self.stopper.set()

    def wait(self, timeout=None):
        if self.real_thread_no_greenlet:
            # we can't '.wait' on this gevent event object, so instead we test it and sleep manually:
            if self.stopper.is_set():
                return True
            non_gevent_sleep(timeout)
            if self.stopper.is_set():
                return True
            return False
        return self.stopper.wait(timeout)

    @contextmanager
    def paused(self):
        self.stop()
        yield
        self.start()

    @contextmanager
    def _running(self):
        if _disabled:
            self._logged_func()
            yield self
            return

        if self.real_thread_no_greenlet:
            _logger.debug('sending job to a real OS thread')
            self._join = defer_to_thread(func=self._logged_func, threadname=self.threadname)
        else:
            # threading.Thread could be a real thread or a gevent-patched thread...
            self.thread = threading.Thread(target=self._logged_func, name=self.threadname, daemon=self.daemon)
            _logger.debug('sending job to %s', self.thread)
            self.stopper.clear()
            self.thread.start()
            self._join = self.thread.join
        try:
            yield self
        finally:
            self.stop()  # if we loop, stop it
        self._join()
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


# re-exports
from .sync import break_locks, TerminationSignal, initialize_exception_listener, initialize_termination_listener, Timebomb
from .sync import set_timebomb, TagAlongThread, SYNC, LoggedRLock, RWLock, SoftLock, skip_if_locked, with_my_lock
from .sync import synchronized, SynchronizedSingleton, LoggedCondition
