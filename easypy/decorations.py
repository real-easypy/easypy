import inspect
import sys
from contextlib import contextmanager
from functools import wraps, partial, update_wrapper
from types import MethodType
import warnings
from threading import RLock
from operator import attrgetter
import weakref
from abc import ABCMeta, abstractmethod

from easypy.collections import intersected_dict, ilistify


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
    """
    Renames arguments while emitting deprecation warning::

        @deprecated_arguments(old_name='new_name')
        def func(new_name):
            # ...

        func(old_name='value meant for new name')
    """

    def wrapper(func):
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
    from .caching import locking_cache

    class CtxManager():
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs
            self.count = 0
            self.func_cm = contextmanager(func)
            self._lock = RLock()

        def __enter__(self):
            with self._lock:
                if self.count == 0:
                    self.ctm = self.func_cm(*self.args, **self.kwargs)
                    self.obj = self.ctm.__enter__()
                self.count += 1
            return self.obj

        def __exit__(self, *args):
            with self._lock:
                self.count -= 1
                if self.count > 0:
                    return
                self.ctm.__exit__(*sys.exc_info())
                inner.cache_pop(*self.args, **self.kwargs)

    @wraps(func)
    @locking_cache
    def inner(*args, **kwargs):
        return CtxManager(*args, **kwargs)

    return inner


class WeakMethodDead(Exception):
    pass


class WeakMethodWrapper:
    def __init__(self, weak_method):
        if isinstance(weak_method, MethodType):
            weak_method = weakref.WeakMethod(weak_method)
        self.weak_method = weak_method
        update_wrapper(self, weak_method(), updated=())
        self.__wrapped__ = weak_method

    def __call__(self, *args, **kwargs):
        method = self.weak_method()
        if method is None:
            raise WeakMethodDead
        return method(*args, **kwargs)


@parametrizeable_decorator
def kwargs_resilient(func, negligible=None):
    """
    If function does not specify **kwargs, pass only params which it can accept

    :param negligible: If set, only be resilient to these specific parameters:

                - Other parameters will be passed normally, even if they don't appear in the signature.
                - If a specified parameter is not in the signature, don't pass it even if there are **kwargs.
    """
    if isinstance(func, weakref.WeakMethod):
        spec = inspect.getfullargspec(inspect.unwrap(func()))
        func = WeakMethodWrapper(func)
    else:
        spec = inspect.getfullargspec(inspect.unwrap(func))
    acceptable_args = set(spec.args or ())
    if isinstance(func, MethodType):
        acceptable_args -= {spec.args[0]}

    if negligible is None:
        @wraps(func)
        def inner(*args, **kwargs):
            if spec.varkw is None:
                kwargs = intersected_dict(kwargs, acceptable_args)
            return func(*args, **kwargs)
    else:
        negligible = set(ilistify(negligible))

        @wraps(func)
        def inner(*args, **kwargs):
            kwargs = {k: v for k, v in kwargs.items()
                      if k in acceptable_args
                      or k not in negligible}
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


@parametrizeable_decorator
def as_list(generator, sort_by=None):
    """
    Forces a generator to output a list.

    When writing a generator is more convenient::

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


class DecoratingDescriptor(metaclass=ABCMeta):
    """
    Base class for descriptors that decorate a function

    :param func: The function to be decorated.
    :param bool cached: If ``True``, the decoration will only be done once per instance.

    Use this as a base class for other descriptors. When used on class objects,
    this will return itself. When used on instances, it this will call ``_decorate``
    on the method created by binding ``func``.
    """

    def __init__(self, *, func, cached: bool):
        self._func = func
        self._cached = cached
        self.__property_name = '__property_%s' % id(self)
        update_wrapper(self, func, updated=())

    @abstractmethod
    def _decorate(self, method, instance, owner):
        """
        Override to perform the actual decoration.

        :param method: The method from binding ``func``.
        :params instance: The binding instance (same as in ``__get__``)
        :params owner: The owner class (same as in ``__get__``)
        """
        pass

    def __get__(self, instance, owner):
        method = self._func.__get__(instance, owner)
        if instance is None:
            return method
        else:
            if self._cached:
                try:
                    return getattr(instance, self.__property_name)
                except AttributeError:
                    bound = self._decorate(method, instance, owner)
                    setattr(instance, self.__property_name, bound)
                    return bound
            else:
                return self._decorate(method, instance, owner)


class LazyDecoratorDescriptor(DecoratingDescriptor):
    def __init__(self, decorator_factory, func, cached):
        super().__init__(func=func, cached=cached)
        self.decorator_factory = decorator_factory

    def _decorate(self, method, instance, owner):
        decorator = self.decorator_factory(instance)
        return decorator(method)


def lazy_decorator(decorator_factory, cached=False):
    """
    Create and apply a decorator only after the method is instantiated::

    :param decorator_factory: A function that will be called with the ``self`` argument.
                              Should return a decorator for the method.
                              If ``string``, use an attribute of ``self`` with that name
                              as the decorator.
    :param bool cached: If ``True``, the decoration will only be done once per instance.

        class UsageWithLambda:
            @lazy_decorator(lambda self: some_decorator_that_needs_the_object(self))
            def foo(self):
                # ...

        class UsageWithAttribute:
            def decorator_method(self, func):
                # ...

            @lazy_decorator('decorator_method')
            def foo(self):
                # ...

        class UsageCached:
            # Without ``cached=True``, this will create a new ``timecache`` on every invocation.
            @lazy_decorator(lambda self: timecache(expiration=1, get_ts_func=lambda: self.ts), cached=True)
            def foo(self):
                # ...
    """

    if callable(decorator_factory):
        pass
    elif isinstance(decorator_factory, str):
        decorator_factory = attrgetter(decorator_factory)
    else:
        raise TypeError('decorator_factory must be callable or string, not %s' % type(decorator_factory))

    def wrapper(func):
        return LazyDecoratorDescriptor(decorator_factory, func, cached)
    return wrapper
