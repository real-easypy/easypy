import inspect
import sys
from contextlib import contextmanager
from functools import wraps, partial
from types import MethodType
import warnings
from threading import RLock

from easypy.collections import intersected_dict


def deprecated(func=None, message=None):
    if not callable(func):
        return partial(deprecated, message=func)
    message = (" "+message) if message else ""
    message = "Hey! '%s' is deprecated!%s" % (func.__name__, message)

    @wraps(func)
    def inner(*args, **kwargs):
        warnings.warn(message, DeprecationWarning, stacklevel=2)
        return func(*args, **kwargs)
    return inner


def parametrizeable_decorator(deco):
    @wraps(deco)
    def inner(func=None, **kwargs):
        if func is None:
            return partial(deco, **kwargs)
        else:
            return wraps(func)(deco(func, **kwargs))
    return inner


def singleton_contextmanager(func):
    class CtxManager():
        def __init__(self, func):
            self.count = 0
            self.func_cm = contextmanager(func)
            self._lock = RLock()

        def __enter__(self):
            with self._lock:
                if self.count == 0:
                    self.ctm = self.func_cm()
                    self.obj = self.ctm.__enter__()
                self.count += 1

        def __exit__(self, *args):
            with self._lock:
                self.count -= 1
                if self.count > 0:
                    return
                self.ctm.__exit__(*sys.exc_info())
                del self.ctm
                del self.obj

    return CtxManager(func)


_singleton_contextmanager_method_attr_lock = RLock()


def singleton_contextmanager_method(func):
    cached_attr_name = '__singleton_contextmanager_method__' + func.__name__

    # Wrap with a context manager to get proper IPython documentation
    @contextmanager
    @wraps(func)
    def inner(self):
        with _singleton_contextmanager_method_attr_lock:
            try:
                cm = getattr(self, cached_attr_name)
            except AttributeError:
                cm = singleton_contextmanager(partial(func, self))
                setattr(self, cached_attr_name, cm)
        with cm as val:
            yield val

    return inner


def kwargs_resilient(func):
    spec = inspect.getfullargspec(getattr(func, '__wrapped__', func))
    acceptable_args = set(spec.args or ())
    if isinstance(func, MethodType):
        acceptable_args -= {spec.args[0]}

    def inner(*args, **kwargs):
        if spec.varkw is None:
            # if function does not get **kwargs, pass only params which it can accept
            kwargs = intersected_dict(kwargs, acceptable_args)
        return func(*args, **kwargs)

    return inner


def reusable_contextmanager(context_manager):
    if not hasattr(context_manager, '_recreate_cm'):
        return context_manager  # context manager is already reusable (was not created usin yield funcion

    class ReusableCtx:
        def __enter__(self):
            self.cm = context_manager._recreate_cm()
            return self.cm.__enter__()

        def __exit__(self, *args):
            self.cm.__exit__(*args)

    return ReusableCtx()


def as_list(generator):
    @wraps(generator)
    def inner(*args, **kwargs):
        return list(generator(*args, **kwargs))
    return inner
