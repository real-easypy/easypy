import sys
from functools import wraps
from easypy.caching import cached_property  # public import, for back-compat


def safe_property(fget=None, fset=None, fdel=None, doc=None):
    """
    A pythonic property which raises a RuntimeError when an attribute error is raised within it. This is
    in order to avoid calls to __getattr__ when properties fail

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
    >>> t.prop #doctest: +IGNORE_EXCEPTION_DETAIL +ELLIPSIS
    Traceback (most recent call last):
     ...
    AssertionError
    >>> t.safe_prop #doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    Traceback (most recent call last):
     ...
    RuntimeError: Attribute error within a property:
    Traceback (...):
     ...
    AttributeError: blap
    """
    if fget is not None:
        @wraps(fget)
        def callable(*args, **kwargs):
            try:
                return fget(*args, **kwargs)
            except AttributeError:
                _, exc, tb = sys.exc_info()
                raise RuntimeError("Attribute error within a property (%s)" % exc).with_traceback(tb)
        return property(callable, fset, fdel, doc)
    else:
        return property(fget, fset, fdel, doc)
