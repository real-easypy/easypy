"""
This module is about making it easier to create decorators
"""

from functools import wraps, partial
from operator import attrgetter


def parametrizeable_decorator(deco):
    @wraps(deco)
    def inner(func=None, **kwargs):
        if func is None:
            return partial(deco, **kwargs)
        else:
            return wraps(func)(deco(func, **kwargs))
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


class LazyDecoratorDescriptor:
    def __init__(self, decorator_factory, func):
        self.decorator_factory = decorator_factory
        self.func = func

    def __get__(self, instance, owner):
        method = self.func.__get__(instance, owner)
        if instance is None:
            return method
        else:
            decorator = self.decorator_factory(instance)
            return decorator(method)


def lazy_decorator(decorator_factory):
    """
    Create and apply a decorator only after the method is instantiated::

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
    """

    if callable(decorator_factory):
        pass
    elif isinstance(decorator_factory, str):
        decorator_factory = attrgetter(decorator_factory)
    else:
        raise TypeError('decorator_factory must be callable or string, not %s' % type(decorator_factory))

    def wrapper(func):
        return LazyDecoratorDescriptor(decorator_factory, func)
    return wrapper
