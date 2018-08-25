"""
This module helps you generate deprecation warnings
"""

from functools import wraps, partial
import warnings


def deprecated(func=None, message=None):
    if not callable(func):
        return partial(deprecated, message=func)
    message = (" " + message) if message else ""
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
