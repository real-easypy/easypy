from __future__ import absolute_import
import inspect
import threading
import functools
from itertools import chain, count
import logging
from weakref import ref
from enum import Enum
from contextlib import contextmanager, ExitStack

from easypy.concurrency import Futures, MultiObject
from easypy.decorations import parametrizeable_decorator, kwargs_resilient
from easypy.exceptions import TException

PRIORITIES = Enum("PRIORITIES", "FIRST NONE LAST")
_logger = logging.getLogger(__name__)


ids = set()


class MissingIdentifier(TException):
    template = "signal handler {_signal_name} must be called with a '{_identifier}' argument"

def make_id(name):
    # from md5 import md5
    # id = md5(name).hexdigest()[:5].upper()
    for c in range(10):
        id = "".join(word[0] for word in name.split("_")) + (str(c) if c else "")
        if id not in ids:
            break
    ids.add(id)
    return id


class SignalHandler(object):

    _idx_gen = count(100)

    def __init__(self, func, async=False, priority=PRIORITIES.NONE, times=None):
        self._func = func  # save the original funtion so we can unregister based on the function
        self.func = kwargs_resilient(func)
        if isinstance(func, functools.partial):
            func = func.func
        self.filename = func.__code__.co_filename
        self.lineno = func.__code__.co_firstlineno
        self.name = self.__name__ = func.__name__
        self.async = async
        self.priority = priority
        self.times = times
        self.idx = next(self._idx_gen)

    def __repr__(self):
        return "<handler #{0.idx} '{0.name}' ({0.filename}:{0.lineno})>".format(self)

    def __call__(self, *, swallow_exceptions, **kwargs):
        if self.times == 0:
            return
        if self.times is not None:
            self.times -= 1
        try:
            return self.func(**kwargs)
        except:
            if not swallow_exceptions:
                raise
            _logger.silent_exception("Exception in (%s) ignored", self)


class Signal:

    ALL = {}

    def __new__(cls, name, async=None, swallow_exceptions=False, log=True):
        try:
            return cls.ALL[name]
        except KeyError:
            pass
        assert threading.main_thread() is threading.current_thread(), "Can only create Signal objects from the MainThread"
        signal = object.__new__(cls)
        # we use this to track the calling of various callbacks. we use this short id so it doesn't overflow the logs
        signal.id = make_id(name)
        signal.handlers = {priority: [] for priority in PRIORITIES}
        signal.name = name
        signal.swallow_exceptions = swallow_exceptions
        signal.async = async
        signal.log = log
        return cls.ALL.setdefault(name, signal)

    def iter_handlers(self):
        return chain(*(self.handlers[k] for k in PRIORITIES))

    def remove_handler(self, handler):
        self.handlers[handler.priority].remove(handler)
        _logger.debug("handler removed from '%s' (%s): %s", self.name, handler.priority.name, handler)

    def register(self, func=None, async=None, priority=PRIORITIES.NONE, times=None):
        if not func:
            return functools.partial(self.register, async=async, priority=priority)
        if async is None:
            async = False if self.async is None else self.async
        elif self.async is not None:
            assert self.async == async, "Signal is set with async=%s" % self.async
        handler = SignalHandler(func, async, priority, times=times)
        self.handlers[priority].append(handler)
        _logger.debug("registered handler for '%s' (%s): %s", self.name, priority.name, handler)
        return func

    def unregister(self, func):
        for handler in self.iter_handlers():
            if handler._func is func:
                self.remove_handler(handler)

    @contextmanager
    def registered(self, func, **kwargs):
        self.register(func, **kwargs)
        try:
            yield
        finally:
            self.unregister(func)

    def __call__(self, **kwargs):
        if self.log:
            # log signal for centralized logging analytics.
            # minimize text message as most of the data is sent in the 'extra' dict
            # signal fields get prefixed with 'sig_', and the values are repr'ed
            signal_fields = {("sig_%s" % k): repr(v) for (k, v) in kwargs.items()}
            _logger.debug("Triggered %s", self, extra=dict(signal_fields, signal=self.name, signal_id=self.id))

        for handler in self.iter_handlers():
            if handler.times == 0:
                self.remove_handler(handler)

        kwargs.setdefault('swallow_exceptions', self.swallow_exceptions)

        with ExitStack() as STACK:
            STACK.enter_context(_logger.context(self.id))

            if any(h.async for h in self.iter_handlers()):
                futures = STACK.enter_context(Futures.executor())
            else:
                futures = None

            for handler in self.iter_handlers():
                # allow handler to use our async context
                handler = _logger.context("#%03d" % handler.idx)(handler)
                if handler.async:
                    futures.submit(handler, **kwargs)
                else:
                    handler(**kwargs)

            if futures:
                for future in futures.as_completed():
                    future.result()        # bubble up exceptions

    def __str__(self):
        return "<Signal %s (%s)>" % (self.name, self.id)

    __repr__ = __str__


def method_identifier_decorator(signal_method, identifier):
    kwargs_resilient_method = kwargs_resilient(signal_method)  # allow the method not to take the identifier as argument
    def new_method(*args, **kwargs):
        self = signal_method.__self__
        if identifier not in kwargs:
            raise MissingIdentifier(_signal_name=signal_method.__name__, _identifier=identifier)

        if hasattr(signal_method, 'identifier_path'):
            path = signal_method.identifier_path
            should_execute = eval('obj.%s' % path.strip('.'), dict(obj=self), {}) == kwargs[identifier]
        else:
            should_execute = kwargs[identifier] in [self, getattr(self, identifier)]
        if not should_execute:
            return
        return kwargs_resilient_method(*args, **kwargs)
    new_method._signal_handler_params = getattr(signal_method, '_signal_handler_params', {})  # inherit handlers params from original method
    return new_method


@parametrizeable_decorator
def signal_identifier_path(func, path):
    func.identifier_path = path
    return func


class ContextManagerSignal(Signal):
    @contextmanager
    def __call__(self, **kwargs):
        if self.log:
            # log signal for centralized logging analytics.
            # minimize text message as most of the data is sent in the 'extra' dict
            # signal fields get prefixed with 'sig_', and the values are repr'ed
            signal_fields = {("sig_%s" % k): repr(v) for (k, v) in kwargs.items()}
            _logger.debug("Triggered '%s' (%s) - entering", self.name, self.id,
                          extra=dict(signal_fields, signal=self.name, signal_id=self.id))

        for handler in self.iter_handlers():
            if handler.times == 0:
                self.unregister(handler.func)

        kwargs.setdefault('swallow_exceptions', self.swallow_exceptions)

        with ExitStack() as handlers_stack:
            async_handlers = MultiObject()
            handlers = []
            for index, handler in enumerate(self.iter_handlers()):
                # allow handler to use our async context
                handler = _logger.context("%02d" % index)(handler)
                handler = _logger.context(self.id)(handler)
                if handler.async:
                    async_handlers.append(handler)
                else:
                    handlers.append(handler)
            if async_handlers:
                handlers_stack.enter_context(async_handlers(**kwargs))
            for handler in handlers:
                res = handler(**kwargs)
                if res:
                    handlers_stack.enter_context(res)
            yield


####################################
def register_signal(name, func, **kwargs):
    signal = Signal(name)
    signal.register(func, **kwargs)
    return functools.partial(signal.unregister, func)


def unregister_signal(name, func):
    try:
        signal = Signal.ALL[name]
    except KeyError:
        return
    return signal.unregister(func)


@parametrizeable_decorator
def register(func, **kwargs):
    register_signal(func.__name__, func, **kwargs)
    return func


def unregister(func):
    return unregister_signal(func.__name__, func)


def _set_handler_params(**kw):
    def inner(func):
        params = getattr(func, '_signal_handler_params', {})
        params.update(kw)
        func._signal_handler_params = params
        return func
    return inner


run_first = _set_handler_params(priority=PRIORITIES.FIRST)
run_last = _set_handler_params(priority=PRIORITIES.LAST)
run_async = _set_handler_params(async=True)
run_sync = _set_handler_params(async=False)


@functools.lru_cache(None)
def get_signals_for_type(typ):
    return {n for typ in inspect.getmro(typ)
            for n in dir(typ) if n.startswith("on_")}


def register_object(obj, **kwargs):
    for method_name in get_signals_for_type(type(obj)):
        method = getattr(obj, method_name)
        assert callable(method), "'%s' (%r) is not callable" % (method, obj)

        # Don't use static/class methods for automatic event registration - they
        # could be registered multiple times if the are multiple objects of that
        # type in the system, or not at all if there are no objects of that type
        # in the system.
        assert method is not getattr(type(obj), method_name, None), "'%s' is a static/class method" % method


        params = getattr(method, '_signal_handler_params', {})
        intersection = set(params).intersection(kwargs)
        assert not intersection, "parameter conflict in signal object registration (%s)" % (intersection)
        params.update(kwargs)
        method_name, *_ = method_name.partition("__")  # allows multiple methods for the same signal

        signal = Signal.ALL.get(method_name)
        if signal and hasattr(signal, 'identifier'):
            method = method_identifier_decorator(method, signal.identifier)

        register_signal(method_name, method, **params)


def unregister_object(obj):
    for method_name in get_signals_for_type(type(obj)):
        method = getattr(obj, method_name[3:])
        if not callable(method):
            continue
        method_name, *_ = method_name.partition("__")
        unregister_signal(method_name, method)


def call_signal(name, **kwargs):
    return Signal(name)(**kwargs)


def log_all_signal_ids(logger=_logger, level=logging.DEBUG):
    for signal in sorted(Signal.ALL.itervalues(), key=lambda signal: signal.name):
        logger.log(level, "%s - '%s'", signal.id, signal.name)


# ===================================================================================================
# Module hack: ``from easypy.signals import on_some_signal``
#              ``from easypy.signals import on_ctx_some_signal`` for a context-manager signal
# ===================================================================================================
def _hack():
    import sys
    from types import ModuleType
    this = sys.modules[__name__]

    class SignalsModule(ModuleType):
        """The module-hack that allows us to use ``from easypy.signals import on_some_signal``"""
        __all__ = ()  # to make help() happy
        __package__ = __name__
        __path__ = []
        __file__ = __file__

        def __getattr__(self, name):
            try:
                return getattr(this, name)
            except AttributeError:
                if name.startswith("on_ctx_"):
                    return ContextManagerSignal(name)
                elif name.startswith("on_"):
                    return Signal(name)
                raise

        def __dir__(self):
            return sorted(dir(this) + list(Signal.ALL))

    module = SignalsModule(__name__, SignalsModule.__doc__)
    sys.modules[module.__name__] = module

_hack()
