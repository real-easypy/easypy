from __future__ import absolute_import

from functools import partial, wraps
from contextlib import contextmanager
from typing import Union
from easypy.timing import Timer
from easypy.units import Duration
import random
import time

from .decorations import parametrizeable_decorator


import logging
from logging import getLogger
_logger = getLogger(__name__)


UNACCEPTABLE_EXCEPTIONS = (NameError, AttributeError, TypeError, KeyboardInterrupt)


class ExponentialBackoff:
    """
    Factory for increasing backoffs.

    :param initial: The value to return on the first call.
    :param base: The ratio between two consecutive returned values.
    :param maximum: A limit to the value increasing.

    Each call generates a backoff (number) that's greater than the previous::

        >>> backoff = ExponentialBackoff(initial=1, base=2)
        >>> backoff()
        1
        >>> backoff()
        2
        >>> backoff()
        4
        >>> backoff()
        8
        >>> backoff()
        16
    """

    def __init__(self, initial=1, maximum=30, base=1.5):
        self.base = base
        self.initial = initial
        self.current = initial
        self.maximum = float(maximum)

    def get_current(self):
        """Get the current value without progressing."""
        return self.current

    def __call__(self):
        """Get the current value and progress to the next one."""
        ret = min(self.get_current(), self.maximum)
        self.current = min(ret * self.base, self.maximum)
        return ret

    def __repr__(self):
        return "{0.__class__.__name__}({0.base}, {0.initial}, {0.maximum} {0.current})".format(self)


class RandomExponentialBackoff(ExponentialBackoff):
    """Variation of ``ExponentialBackoff`` that adds randomization to the returned values."""

    def get_current(self):
        return random.random() * self.current + self.initial


class ExpiringCounter(object):
    """
    A counter that decreases every time its expiration is checked.

    >>> counter = ExpiringCounter(3)
    >>> counter.expired
    False
    >>> counter.expired
    False
    >>> counter.expired
    False
    >>> counter.expired
    True
    """
    def __init__(self, times):
        self.times = times

    @property
    def expired(self):
        """Progress the counter and return whether or not it expired."""
        self.times -= 1
        return self.times < 0

    @property
    def remain(self):
        """Return the number of remaining times to check ``expired`` before it expires."""
        return self.times


def retry(times: Union[int, Duration, Timer], func, args=[], kwargs={}, acceptable=Exception, sleep=1,
          max_sleep=False, log_level=logging.DEBUG, pred=None, unacceptable=()):
    """
    Runs a function again and again until it finishes without throwing.

    :param times: Limit number of attempts before errors are propagated instead of suppressed.
                  If an `int`, the execution is retried at most `times` times.
                  If a `Timer`, retries until the `times` expires.
                  If a `Duration`, retires for the specified duarion.
    :param func: The function to run.
    :param list args: Positional arguments for the function.
    :param dict kwargs: Keyword arguments for the function.
    :param acceptable: Exception (or tuple of exceptions) to repeat the function on.
    :param sleep: Time to wait between multiple retries.
    :param max_sleep: If set, increase the sleep after each retry until reaching this number.
    :param log_level: Before retrying, log the exceptions at that level.
    :param pred: If set, decide whether or not to retry on an exception.
    :param unacceptable: Exception (or tuple of exceptions) to not repeat the function on and re-raise them instead.
    """

    if unacceptable is None:
        unacceptable = ()
    elif isinstance(unacceptable, tuple):
        unacceptable += UNACCEPTABLE_EXCEPTIONS
    else:
        unacceptable = (unacceptable,) + UNACCEPTABLE_EXCEPTIONS

    if isinstance(times, Timer):
        stopper = times  # a timer is a valid stopper
    elif isinstance(times, Duration):
        stopper = Timer(expiration=times)
    elif isinstance(times, int):
        stopper = ExpiringCounter(times)
    else:
        assert False, "'times' must be an 'int', 'Duration' or 'Timer', got %r" % times

    if max_sleep:
        sleep = RandomExponentialBackoff(sleep, max_sleep)
    if not pred:
        pred = lambda exc: True

    while True:
        try:
            return func(*args, **kwargs)
        except unacceptable as exc:
            raise
        except acceptable as exc:
            raise_if_async_exception(exc)
            if not pred(exc):
                raise
            if stopper.expired:
                raise
            _logger.log(log_level, "Exception thrown: %r", exc)
            sleep_for = 0
            if sleep:
                # support for ExponentialBackoff
                sleep_for = sleep() if callable(sleep) else sleep
            _logger.log(log_level, "Retrying... (%r remain) in %s seconds", stopper.remain, sleep_for)
            time.sleep(sleep_for)


def retrying(times: Union[int, Duration, Timer], acceptable=Exception, sleep=1, max_sleep=False, log_level=logging.DEBUG, pred=None, unacceptable=()):
    """Try running the decorated function, retrying if an acceptable exception caught.

    times - limit number of attempts before errors are propagated instead of suppressed.
            if an `int`, the execution is retried at most `times` times.
            if a `Timer`, retries until the `times` expires.
            if a `Duration`, retries up until `Timer(times)` expires
    acceptable - exception or tuple of exceptions which to catch and retry upon.
    sleep - time to wait between attempts. can be a callable.
    max_sleep - if given, then the time to sleep between attempts is `RandomExponentialBackoff(initial=sleep, maximum=max_sleep)`.
    log_level - level of the log to emit when catching an exception and retrying.
    pred - if given, then retries only if pred(exception) is True.

    >>> @retrying(times=5)
    ... def get_lucky():
    ...     if random.random() < 0.5:
    ...         raise Exception('No luck')
    ...     return 'Got lucky'
    """
    def wrapper(func):
        @wraps(func)
        def impl(*args, **kwargs):
            return retry(
                times, func, args, kwargs, sleep=sleep, max_sleep=max_sleep, acceptable=acceptable,
                log_level=log_level, pred=pred, unacceptable=unacceptable)
        return impl
    return wrapper


class _Retry(Exception):
    pass


retrying.Retry = _Retry

retrying.debug = partial(retrying, log_level=logging.DEBUG)
retrying.info = partial(retrying, log_level=logging.INFO)
retrying.warning = partial(retrying, log_level=logging.WARNING)
retrying.error = partial(retrying, log_level=logging.ERROR)


@parametrizeable_decorator
def resilient(func=None, default=None, **kw):
    """Suppress exceptions raised from the decorated function.

    default - value to return if an exception is suppressed.
    kw      - see `resilience`

    >>> @resilient(default=0)
    ... def get_number():
    ...     return int(open('non-existent-file').read())
    >>> get_number()
    0
    """
    kw.setdefault('msg', "ignoring error in %s ({type})" % func.__qualname__)

    @wraps(func)
    def inner(*args, **kwargs):
        with resilience(**kw):
            return func(*args, **kwargs)
        return default  # we reach here only if an exception was caught and handled by resilience
    return inner


@contextmanager
def resilience(msg="ignoring error {type}", acceptable=Exception, unacceptable=(), log_level=logging.DEBUG, pred=None):
    """Suppress exceptions raised from the wrapped scope.

    msg          - format of log to print when an exception is suppressed.
    acceptable   - exception or tuple of exceptions which to suppress.
    unacceptable - exception or tuple of exception which to not suppress, even if they are in `acceptable`.
                   the exceptions in UNACCEPTABLE_EXCEPTIONS are always unacceptable, unless `unacceptable` is None.
    log_level    - level of the log to emit when suppressing an exception.
    pred         - if given, then an exception is suppressed only if pred(exception) is True.

    >>> import errno
    >>> with resilience(acceptable=OSError, pred=lambda ex: ex.errno == errno.ENOENT):
    ...     print('before')
    ...     open('non-existent-file')
    ...     print('after')
    before
    """
    if unacceptable is None:
        unacceptable = ()
    elif isinstance(unacceptable, tuple):
        unacceptable += UNACCEPTABLE_EXCEPTIONS
    else:
        unacceptable = (unacceptable,) + UNACCEPTABLE_EXCEPTIONS
    try:
        yield
    except unacceptable as exc:
        raise
    except acceptable as exc:
        if pred and not pred(exc):
            raise
        raise_if_async_exception(exc)
        _logger.log(log_level, msg.format(exc=exc, type=exc.__class__.__qualname__))
        if log_level > logging.DEBUG:
            _logger.debug("Traceback:", exc_info=True)


resilient.debug = partial(resilient, log_level=logging.DEBUG)
resilient.info = partial(resilient, log_level=logging.INFO)
resilient.warning = partial(resilient, log_level=logging.WARNING)
resilient.error = partial(resilient, log_level=logging.ERROR)

resilience.debug = partial(resilience, log_level=logging.DEBUG)
resilience.info = partial(resilience, log_level=logging.INFO)
resilience.warning = partial(resilience, log_level=logging.WARNING)
resilience.error = partial(resilience, log_level=logging.ERROR)


def raise_if_async_exception(exc):
    """
    Helper for handling exceptions potentially invoked in another thread.

    This is so that exception raised from other threads don't get supressed by retry/resilience.
    See :meth:`~easypy.concurrency.raise_in_main_thread`.
    """
    if getattr(exc, "_raised_asynchronously", False):
        _logger.info('Raising asynchronous error')
        raise exc
