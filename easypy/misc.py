import inspect
import weakref
from functools import wraps, update_wrapper
from types import MethodType
from .decorations import parametrizeable_decorator
from .collections import ilistify, intersected_dict
from .tokens import Token  # for backwards compatibility


class Hex(int):

    def __str__(self):
        return "%X" % self

    def __repr__(self):
        return "0x%x" % self


def get_all_subclasses(cls, include_mixins=False):

    def is_mixin(subclass):
        return getattr(subclass, "_%s__is_mixin" % subclass.__name__, False)

    def gen(cls):
        for subclass in cls.__subclasses__():
            if include_mixins or not is_mixin(subclass):
                yield subclass
            yield from gen(subclass)

    return list(gen(cls))


def stack_level_to_get_out_of_file():
    frame = inspect.currentframe().f_back
    filename = frame.f_code.co_filename
    stack_levels = 1
    while frame.f_code.co_filename == filename:
        stack_levels += 1
        frame = frame.f_back
    return stack_levels


def at_most(val, mx_val):
    return min(val, mx_val)


def at_least(val, mn_val):
    return max(val, mn_val)


def clamp(val, at_least, at_most):
    """
    Clamps a value so it doesn't exceed specified limits.
    If one of the edges is not needed, it should be passed as None (consider using at_most / at_least functions).
    :param at_least: Minimum possible value.
    :param at_most: Maxium possible value.
    :return: The clamped value.
    """

    if at_least > at_most:
        raise ValueError("Min value cannot be higher than max value.")

    if at_most is not None:
        val = min(at_most, val)
    if at_least is not None:
        val = max(at_least, val)
    return val


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
