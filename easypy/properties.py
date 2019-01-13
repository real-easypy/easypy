"""
This module is about 'property' descriptors.
"""

import sys
from functools import wraps
from easypy.caching import cached_property  # public import, for back-compat


_builtin_property = property


def safe_property(fget=None, fset=None, fdel=None, doc=None):
    """
    A pythonic property which raises a RuntimeError when an attribute error is raised within it.
    This fixes an issue in python where AttributeErrors that occur anywhere _within_ 'property' functions
    are effectively suppressed, and converted to AttributeErrors for the property itself. This is confusing
    for the debugger, and also leads to unintended fallback calls to a __getattr__ if defined

    >>> def i_raise_an_exception():
    ...     raise AttributeError("blap")

    >>> class Test(object):
    ...     def some_prop(self):
    ...         return i_raise_an_exception()
    ...     def __getattr__(self, attr):
    ...         assert False
    ...     prop = property(some_prop)
    ...     safe_prop = safe_property(some_prop)
    >>> t = Test()
    >>> t.prop
    Traceback (most recent call last):
     ...
    AssertionError
    >>> t.safe_prop
    Traceback (most recent call last):
     ...
     AttributeError: blap
     ...
     During handling of the above exception, another exception occurred:
     ...
     Traceback (most recent call last):
     ...
    RuntimeError: Attribute error within a property (blap)
    """
    if fget is not None:
        @wraps(fget)
        def callable(*args, **kwargs):
            try:
                return fget(*args, **kwargs)
            except AttributeError:
                _, exc, tb = sys.exc_info()
                raise RuntimeError("Attribute error within a property (%s)" % exc).with_traceback(tb)
        return _builtin_property(callable, fset, fdel, doc)
    else:
        return _builtin_property(fget, fset, fdel, doc)
