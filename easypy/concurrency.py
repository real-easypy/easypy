# encoding: utf-8

"""
This module helps you run things concurrently.
Most useful are the ``concurrent`` context manager and the ``MultiObject`` class.
The features in this module are integrated with the ``logging`` module, to provide
thread-context to log messages. It also has support for integration ``gevent``.

``MultiObject``

Here's how you would, for example, concurrently send a message to a bunch of servers::

    responses = MultiObject(servers).send('Hello')

The above replaces the following threading boiler-plate code::

    from threading import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(server.send, args=['Hello'])
            for server in servers]

    responses = [future.result() for future in futures]```


``concurrent``

A high-level thread controller.
As a context manager, it runs a function in a thread, and joins it upon exiting the context

    with concurrent(requests.get, "api.myserver.com/data") as future:
        my_data = open("local_data").read()

    remote_data = future.result()
    process_it(my_data, remote_data)

It can also be used to run something repeatedly in the background:

    concurrent(send_heartbeat, loop=True, sleep=5).start()  # until the process exits

    with concurrent(print_status, loop=True, sleep=5):
        some_long_task()


Environment variables

    EASYPY_DISABLE_CONCURRENCY (yes|no|true|false|1|0)
    EASYPY_MAX_THREAD_POOL_SIZE

Important notes

    * Exception.timestamp

    Exceptions raised in functions that use the features here may get a ``timestamp`` attribute that
    records the precise time those exceptions were raised. This is useful when there's a lag between
    when the exception was raised within a thread, and when that exception was finally propagated to
    the calling thread.

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
import os
from collections import namedtuple
from datetime import datetime
from unittest.mock import MagicMock

import easypy._multithreading_init  # noqa; make it initialize the threads tree
from easypy.exceptions import PException
from easypy.gevent import is_module_patched, non_gevent_sleep, defer_to_thread
from easypy.humanize import IndentableTextBuffer, time_duration, compact
from easypy.humanize import format_thread_stack, yesno_to_bool
from easypy.threadtree import iter_thread_frames
from easypy.timing import Timer
from easypy.units import MINUTE, HOUR
from easypy.colors import colorize, uncolored
from easypy.sync import SynchronizationCoordinator, ProcessExiting, raise_in_main_thread


MAX_THREAD_POOL_SIZE = int(os.environ.get('EASYPY_MAX_THREAD_POOL_SIZE', 50))
DISABLE_CONCURRENCY = yesno_to_bool(os.getenv("EASYPY_DISABLE_CONCURRENCY", "no"))

this_module = import_module(__name__)
THREADING_MODULE_PATHS = [threading.__file__]


if is_module_patched("threading"):
    import gevent
    MAX_THREAD_POOL_SIZE *= 100  # these are not threads anymore, but greenlets. so we allow a lot of them
    THREADING_MODULE_PATHS.append(gevent.__path__[0])


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


def disable():
    """
    Force MultiObject and concurrent calls to run synchronuously in the current thread.
    For debugging purposes.
    """
    global DISABLE_CONCURRENCY
    DISABLE_CONCURRENCY = True
    logging.info("Concurrency disabled")


def enable():
    """
    Re-enable concurrency, after disabling it
    """
    global DISABLE_CONCURRENCY
    DISABLE_CONCURRENCY = False
    logging.info("Concurrency enabled")


def _find_interesting_frame(f):
    """
    Find the next frame in the stack that isn't threading-related, to get to the actual caller.
    """
    default = next(_extract_stack_iter(f))
    non_threading = (
        p for p in _extract_stack_iter(f)
        if all(not p[0].startswith(pth) for pth in THREADING_MODULE_PATHS))
    return next(non_threading, default)


class MultiExceptionMeta(type):
    """
    This metaclass helps generate MultiException subtypes so that it's easier
    to catch a MultiException with a specific common type, for example::

        try:
            MultiObject(servers).connect()
        except MultiException[ConnectionError]:
            pass

    See ``MultiException`` for more information. 
    """

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
    """
    A ``MultiException`` subtype is raised when a ``MultiObject`` call fails in one or more of its threads.
    The exception contains the following members:

    :param actual: a MultiObject of all exceptions raised in the ``MultiObject`` call
    :param count: the number of threads that raised an exception
    :param invocations_count: the total number of calls the ``MultiObject`` made (the size of the ``MultiObject``)
    :param common_type: the closest common ancestor (base-class) of all the exceptions
    :param one: a sample exception (the first)
    :param futures: a MultiObject of futures (:concurrent.futures.Future:) that were created in the ``MultiObject`` call
    :param exceptions: a sparse list of exceptions corresponding to the MultiObject threads
    :param complete: ``True`` if all threads failed on exception
    """

    template = "{0.common_type.__qualname__} raised from concurrent invocation (x{0.count}/{0.invocations_count})"

    def __reduce__(self):
        return (MultiException, (self.exceptions, [PickledFuture(ctx=f.ctx, funcname=f.funcname) for f in self.futures]))

    def __init__(self, exceptions, futures):
        # we want to keep futures in parallel with exceptions,
        # so some exceptions could be None
        assert len(futures) == len(exceptions)
        self.actual = MultiObject(filter(None, exceptions))
        self.count = len(self.actual)
        self.invocations_count = len(futures)
        self.common_type = self.COMMON_TYPE
        self.one = self.actual.T[0] if self.actual else None
        self.futures = MultiObject(futures)
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
        return colorize("\n" + text)

    def _get_buffer(self, **kw):
        if kw.get("color", True):
            normalize_color = lambda x: x
        else:
            normalize_color = uncolored

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


def _submit_execution(executor, func, args, kwargs, ctx, funcname=None):
    """
    This helper takes care of submitting a function for asynchronous execution, while wrapping and storing
    useful information for tracing it in logs (for example, by ``Futures.dump_stacks``)
    """
    future = executor.submit(_run_with_exception_logging, func, args, kwargs, ctx)
    future.ctx = ctx
    future.funcname = funcname or _get_func_name(func)
    return future


class Futures(list):
    """
    A collection of ``Future`` objects.
    """

    def done(self):
        """
        Return ``True`` if all futures are done
        """
        return all(f.done() for f in self)

    def cancelled(self):
        """
        Return ``True`` if all futures are cancelled
        """
        return all(f.cancelled() for f in self)

    def running(self):
        """
        Return ``True`` if all futures are running
        """
        return all(f.running() for f in self)

    def wait(self, timeout=None):
        """
        Wait for all Futures to complete
        """
        return futures_wait(self, timeout=timeout)

    def result(self, timeout=None):
        """
        Wait and return the results from all futures as an ordered list.
        Raises a ``MultiException`` if one or more exceptions are raised.
        """
        me = self.exception(timeout=timeout)
        if me:
            if isinstance(me, MultiException[ProcessExiting]):
                # we want these aborted MultiObject threads to consolidate this exception
                raise ProcessExiting()
            raise me
        return [f.result() for f in self]

    def exception(self, timeout=None):
        """
        Wait and return a ``MultiException`` if there any exceptions, otherwise returns ``None``.
        """
        exceptions = [f.exception(timeout=timeout) for f in self]
        if any(exceptions):
            return MultiException(exceptions=exceptions, futures=self)

    def cancel(self):
        """
        Cancel all futures.
        """
        cancelled = [f.cancel() for f in self]  # list-comp, to ensure we call cancel on all futures
        return all(cancelled)

    def as_completed(self, timeout=None):
        """
        Returns an iterator yielding the futures in order of completion.
        Wraps `concurrent.futures.as_completed`.
        """
        return as_completed(self, timeout=timeout)

    @classmethod
    @contextmanager
    def execution(cls, workers=None, ctx={}):
        """
        A context-manager for scheduling asynchronous executions and waiting on them as upon exiting the context::

            With Futures.execution() as futures:
                for task in tasks:
                    futures.submit(task)

            results = futures.results()
        """
        if workers is None:
            workers = MAX_THREAD_POOL_SIZE

        class PooledFutures(cls):

            killed = False

            def submit(self, func, *args, log_ctx={}, **kwargs):
                "Submit a new asynchronous task to this executor"

                _ctx = dict(ctx, **log_ctx)
                future = executor.submit(_run_with_exception_logging, func, args, kwargs, _ctx)
                future.ctx = _ctx
                future.funcname = _get_func_name(func)
                self.append(future)
                return future

            def kill(self):
                "Kill the executor and discard any running tasks"
                self.cancel()
                self.shutdown(wait=False)
                while executor._threads:
                    thread = executor._threads.pop()
                    if getattr(thread, "_greenlet", None):
                        thread._greenlet.kill()
                self.killed = True

            def shutdown(self, *args, **kwargs):
                executor.shutdown(*args, **kwargs)

        with ThreadPoolExecutor(workers) as executor:

            futures = PooledFutures()

            try:
                yield futures
            except:  # noqa
                _logger.debug("shutting down ThreadPoolExecutor due to exception")
                futures.kill()
                raise
            else:
                if not futures.killed:
                    # force exceptions to bubble up
                    futures.result()
            finally:
                # break the cycle so that the GC doesn't clean up the executor under a lock (https://bugs.python.org/issue21009)
                futures.kill = futures.shutdown = futures.submit = None
                futures = None

    executor = execution  # for backwards compatibility

    @classmethod
    def dump_stacks(cls, futures, verbose=False):
        """
        Logs the stack frame for each of the given futures.
        The Future objects must have been submitted with ``_submit_execution`` so that they contain
        the necessary information.
        """
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
                             i, future.funcname, cls._get_context(future), location)
                continue

            with _logger.indented("%3s - %s (%s)", i, future.funcname, cls._get_context(future), footer=False):
                lines = format_thread_stack(frame, skip_modules=[this_module]).splitlines()
                for line in lines:
                    _logger.info(line.strip())

    @classmethod
    def _get_context(cls, future: Future):
        """
        Get interesting context information about this future object (as long as it was submitted by _submit_execution)
        """
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

    def logged_wait(self, timeout=None, initial_log_interval=2 * MINUTE):
        """
        Wait for all futures to complete, logging their status along the way.
        Logging will occur at an every-increasing log interval, beginning with ``initial_log_interval``,
        and increasing 5-fold (x5) every 5 iterations.
        """
        log_interval = initial_log_interval
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
    """
    Use as a wrapper for functions that run asynchronously, setting up a logging context and
    recording the thread in-which they are running, so that we can later log their progress
    and identify the source of exceptions they raise. In addition, it stamps any exception
    raised from the function with the current time.
    """
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
    "Helper for normalizing a list of parameters to be mapped on a function"
    # We use type(args) == tuple because isinstance will return True for namedtuple
    return [args if type(args) == tuple else (args,) for args in params]


def _get_func_name(func):
    "Helper for finding an appropriate name for the given callable, handling ``partial`` objects."
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
    "Helper for normalizing a list of parameters and log-contexts into a list of usable log context dicts"
    if not log_contexts:
        log_contexts = (dict(context=str(p) if len(p) > 1 else str(p[0])) for p in params)
    else:
        log_contexts = (p if isinstance(p, dict) else dict(context=str(p))
                        for p in log_contexts)
    return log_contexts


@contextmanager
def asynchronous(func, params=None, workers=None, log_contexts=None, final_timeout=2.0, **kw):
    """
    Map the list of tuple-parameters onto asynchronous calls to the specified function::

        with asynchronous(connect, [(host1,), (host2,), (host3,)]) as futures:
            ...

        connections = futures.results()

    :param func: The callable to invoke asynchronously.
    :param params: A list of tuples to map onto the function.
    :param workers: The number of workers to use. Defaults to the number of items in ``params``.
    :param log_contexts: A optional list of logging context objects, matching the items in ``params``.
    :param final_timeout: The amount of time to allow for the futures to complete after exiting the asynchronous context.
    """
    if params is None:
        params = [()]

    if not isinstance(params, list):  # don't use listify - we want to listify tuples too
        params = [params]

    params = _to_args_list(params)
    log_contexts = _to_log_contexts(params, log_contexts)
    if workers is None:
        workers = min(MAX_THREAD_POOL_SIZE, len(params))

    try:
        signature = inspect.signature(func)
    except ValueError:
        # In Python 3.5+, inspect.signature returns this for built-in types
        pass
    else:
        if '_sync' in signature.parameters and '_sync' not in kw:
            assert len(params) <= workers, 'SynchronizationCoordinator with %s tasks but only %s workers' % (len(params), workers)
            synchronization_coordinator = SynchronizationCoordinator(len(params))
            kw['_sync'] = synchronization_coordinator

            func = synchronization_coordinator._abandon_when_done(func)

    if not params:
        # noop
        yield Futures()
        return

    with Futures.executor(workers=workers) as futures:
        for args, ctx in zip(params, log_contexts):
            futures.submit(func, *args, log_ctx=ctx, **kw)

        yield futures


def concurrent_find(func, params, **kw):
    assert not DISABLE_CONCURRENCY, "concurrent_find runs only with concurrency enabled"
    timeout = kw.pop("concurrent_timeout", None)
    with asynchronous(func, list(params), **kw) as futures:
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
    if DISABLE_CONCURRENCY or len(params) == 1:
        return nonconcurrent_map(func, params, log_contexts, **kw)

    with asynchronous(func, list(params), workers, log_contexts, **kw) as futures:
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
    """
    Higher-level thread execution.

    :param func: The callable to invoke asynchronously.
    :param throw: When used as a context-manager, if an exception was thrown inside the thread, re-raise it to the calling thread upon exiting the context. (default: True)
    :param daemon: Set the thread as daemon, so it does not block the process from exiting if it did not complete. (default: True)
    :param threadname: Set a name for this thread. (default: ``anon-<id>``)

    :param loop: If ``True``, repeatedly calls ``func`` until the context is exited, ``.stop()`` is called, or the ``stopper`` event object is set. (default: False)
    :param sleep: Used with the ``loop`` flag - the number of seconds between consecutive calls to ``func``. (default: 1)
    :param stopper: Used with the ``loop`` flag - an external ``threading.Event`` object to use for stopping the loop .

    :param console_logging: If ``False``, suppress logging to the console log handler. (default: False)

    Running multiple tasks side-by-side::

        with \
            concurrent(requests.get, "api.myserver.com/data1") as async1, \
            concurrent(requests.get, "api.myserver.com/data2") as async2:
            my_data = open("local_data").read()

        remote_data1 = async1.result()
        remote_data2 = async2.result()
        process_it(my_data, remote_data1, remote_data2)

    Run something repeatedly in the background:

        heartbeats = concurrent(send_heartbeat, loop=True, sleep=5)
        heartbeats.start()  # until stopped, or the process exits

        with concurrent(print_status, loop=True, sleep=5):
            some_long_task()

        heartbeats.stop()
    """

    def __init__(self, func, *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.throw = kwargs.pop('throw', True)
        self.daemon = kwargs.pop('daemon', True)
        self.stopper = kwargs.pop('stopper', threading.Event())
        self.sleep = kwargs.pop('sleep', 1)
        self.loop = kwargs.pop('loop', False)
        self.timer = None
        self.console_logging = kwargs.pop('console_logging', True)
        self.threadname = kwargs.pop('threadname', None)
        if not self.threadname:
            current_thread_name = threading.current_thread().name
            if current_thread_name:
                current_thread_name = current_thread_name.split('::')[0]  # We want to see only the context
                self.threadname = "%s::%X" % (current_thread_name, id(self))
            else:
                self.threadname = "anon-%X" % id(self)

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
                self._result = self.func(*self.args, **self.kwargs)
                if not self.loop:
                    return
                if self.wait(self.sleep):
                    _logger.debug("%s - stopped", self)
                    return
        except ProcessExiting as exc:
            _logger.debug(exc)
            raise
        except (KeyboardInterrupt, Exception) as exc:
            _logger.silent_exception("Exception in thread running %s: %s (traceback can be found in debug-level logs)", self.func, type(exc))
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

    def result(self, timeout=None):
        self.wait(timeout=timeout)
        if self.throw and self.exc:
            raise self.exc
        return self._result

    def done(self):
        """
        Return ``True`` if the thread is done (successfully or not)
        """
        return hasattr(self, '_result') or getattr(self, 'exc', None) is not None

    @contextmanager
    def paused(self):
        self.stop()
        yield
        self.start()

    @contextmanager
    def _running(self):
        if DISABLE_CONCURRENCY:
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
        # TODO: document or remove
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
from .sync import synchronized, SynchronizedSingleton, LoggedCondition, _check_exiting


import sys
if sys.version_info < (3, 7):
    # async became reserved in 3.7, but we'll keep it for backwards compatibility
    code = compile("async = asynchronous", __name__, "exec")
    eval(code, globals(), globals())
