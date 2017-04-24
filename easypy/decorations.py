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


def deprecated_arguments(**argmap):
    def wrapper(func):
        """
        Renames arguments while emitting deprecation warning

        @deprecated_arguments(old_name='new_name')
        def func(new_name):
            # ...

        func(old_name='value meant for new name')
        """

        @wraps(func)
        def inner(*args, **kwargs):
            deprecation_warnings = []
            for name, map_to in argmap.items():
                try:
                    value = kwargs.pop(name)
                except KeyError:
                    pass  # deprecated argument was not used
                else:
                    if map_to in kwargs:
                        raise TypeError("%s is deprecated for %s - can't use both in %s()" % (
                                        name, map_to, func.__name__))
                    deprecation_warnings.append('%s is deprecated - use %s instead' % (name, map_to))
                    kwargs[map_to] = value

            if deprecation_warnings:
                message = 'Hey! In %s, %s' % (func.__name__, ', '.join(deprecation_warnings))
                warnings.warn(message, DeprecationWarning, stacklevel=2)

            return func(*args, **kwargs)
        return inner
    return wrapper


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


def kwargs_as_needed(func):
    """
    If function does not specify **kwargs, pass only params which it can accept
    """

    spec = inspect.getfullargspec(getattr(func, '__wrapped__', func))
    acceptable_args = set(spec.args or ())
    if isinstance(func, MethodType):
        acceptable_args -= {spec.args[0]}

    @wraps(func)
    def inner(*args, **kwargs):
        if spec.varkw is None:
            kwargs = intersected_dict(kwargs, acceptable_args)
        return func(*args, **kwargs)

    return inner


kwargs_resilient = kwargs_as_needed


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


@parametrizeable_decorator
def as_list(generator, sort_by=None):
    """
    Forces a generator to output a list.
    When writing a generator is more convenient

    @as_list(sort_by=lambda n: -n)
    def g():
        yield 1
        yield 2
        yield from range(2)

    >>> g()
    [2, 1, 1, 0]

    """
    @wraps(generator)
    def inner(*args, **kwargs):
        l = list(generator(*args, **kwargs))
        if sort_by:
            l.sort(key=sort_by)
        return l
    return inner
