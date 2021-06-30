"""
This module is about making it easier to create decorators
"""

from collections import OrderedDict
from functools import wraps, partial, update_wrapper
from itertools import chain
from operator import attrgetter
from abc import ABCMeta, abstractmethod
import inspect

from easypy.exceptions import TException


def parametrizeable_decorator(deco):
    @wraps(deco)
    def inner(func=None, *args, **kwargs):
        if func is None:
            return partial(deco, *args, **kwargs)
        else:
            return wraps(func)(deco(func, *args, **kwargs))
    return inner


def wrapper_decorator(deco):
    @wraps(deco)
    def inner(func):
        return wraps(func)(deco(func))
    return inner


def reusable_contextmanager(context_manager):
    """
    Allows the generator-based context manager to be used more than once
    """

    if not hasattr(context_manager, '_recreate_cm'):
        return context_manager  # context manager is already reusable (was not created usin yield funcion

    class ReusableCtx:
        def __enter__(self):
            self.cm = context_manager._recreate_cm()
            return self.cm.__enter__()

        def __exit__(self, *args):
            self.cm.__exit__(*args)

    return ReusableCtx()


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
                    return instance.__dict__[self.__property_name]
                except KeyError:
                    bound = self._decorate(method, instance, owner)
                    instance.__dict__[self.__property_name] = bound
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


class DefaultsMismatch(TException):
    template = 'The defaults of {func} differ from those of {source_of_truth} in params {param_names}'


def ensure_same_defaults(source_of_truth, ignore=()):
    """
    Ensure the decorated function has the same default as the source of truth in optional parameters shared by both

    :param source_of_truth: A function to check the defaults against.
    :param ignore: A list of parameters to ignore even if they exist and have defaults in both functions.
    :raises DefaultsMismatch: When the defaults are different.

    >>> def foo(a=1, b=2, c=3):
    ...     ...
    >>> @ensure_same_defaults(foo)
    ... def bar(a=1, b=2, c=3):  # these defaults are verified by the decorator
    ...         ...
    """

    sot_signature = inspect.signature(source_of_truth)
    params_with_defaults = [
        param for param in sot_signature.parameters.values()
        if param.default is not param.empty
        and param.name not in ignore]

    def gen_mismatches(func):
        signature = inspect.signature(func)
        for sot_param in params_with_defaults:
            param = signature.parameters.get(sot_param.name)
            if param is None:
                continue
            if param.default is param.empty:
                continue
            if sot_param.default != param.default:
                yield sot_param.name

    def wrapper(func):
        mismatches = list(gen_mismatches(func))
        if mismatches:
            raise DefaultsMismatch(
                func=func,
                source_of_truth=source_of_truth,
                param_names=mismatches)
        return func
    return wrapper


__KEYWORD_PARAMS = (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)


def kwargs_from(*functions, exclude=()):
    """
    Edits the decorated function's signature to expand the variadic keyword
    arguments parameter to the possible keywords from the wrapped functions.
    This allows better completions inside interactive tools such as IPython.

    :param functions: The functions to get the keywords from.
    :param exclude: A list of parameters to exclude from the new signature.
    :raises TypeError: When the decorated function does not have a variadic
                       keyword argument.

    >>> def foo(*, a, b, c):
    ...     ...
    >>> @kwargs_from(foo)
    ... def bar(**kwargs):
    ...     ...
    >>> help(bar)
    Help on function bar in module easypy.decorations:
    <BLANKLINE>
    bar(*, a, b, c)
    <BLANKLINE>
    """
    exclude = set(exclude or ())
    all_original_params = (inspect.signature(func).parameters for func in functions)
    def _decorator(func):
        signature = inspect.signature(func)

        kws_param = None
        params = OrderedDict()
        for param in signature.parameters.values():
            if param.kind != inspect.Parameter.VAR_KEYWORD:
                params[param.name] = param
            else:
                kws_param = param
        if kws_param is None:
            raise TypeError("kwargs_from can only wrap functions with variadic keyword arguments")

        keep_kwargs = False
        for param in chain.from_iterable(original_params.values() for original_params in all_original_params):
            if param.name in exclude:
                pass
            elif param.kind in __KEYWORD_PARAMS and param.name not in params:
                params[param.name] = param.replace(kind=inspect.Parameter.KEYWORD_ONLY)
            elif param.kind == inspect.Parameter.VAR_KEYWORD:
                keep_kwargs = True

        if keep_kwargs:
            params['**'] = kws_param

        func.__signature__ = signature.replace(parameters=params.values())
        return func
    return _decorator
