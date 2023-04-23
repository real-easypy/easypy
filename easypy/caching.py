import os
import sys
import time
import shelve
import inspect
from threading import RLock
from collections import defaultdict
from contextlib import ExitStack, contextmanager
from functools import wraps, _make_key, partial, lru_cache, update_wrapper

from .logging import DeferredEasypyLogger
from .decorations import parametrizeable_decorator, DecoratingDescriptor
from .collections import ilistify
from .misc import kwargs_resilient
from .deprecation import deprecated
from .units import HOUR
from .tokens import DELETED, NO_DEFAULT
from .humanize import yesno_to_bool

_logger = DeferredEasypyLogger(name=__name__)

try:
    from _gdbm import error as GDBMException
except:  # noqa
    try:
        from _dbm import error as GDBMException
    except:  # noqa
        from dbm import error as GDBMException


DISABLE_CACHING_PERSISTENCE = yesno_to_bool(os.getenv("EASYPY_DISABLE_CACHING_PERSISTENCE", "no"))


class PersistentCache(object):
    """
    A memoizer that stores its cache persistantly using shelve.

    :param path: location of cache shelve file.
    :param version: modify to deprecate old cahed data
    :param expiration: expiration in seconds for the entire cache. ``None`` to disable expiration.

    Example::

        >>> CACHE = PersistentCache("/tmp/cache", version=1, expiration=60)

        >>> @CACHE
        ... def fib(n):
        ...     a, b = 0, 1
        ...     while a < n:
        ...         a, b = b, a+b
        ...     return a
    """

    def __init__(self, path, version=None, expiration=4 * HOUR, ignored_keywords=None):
        self.path = path
        self.version = version
        if version is not None:
            self.path = "{self.path}.v{version}".format(**locals())
        self.expiration = expiration
        self.lock = RLock()
        self.ignored_keywords = set(ilistify(ignored_keywords)) if ignored_keywords else set()

    @contextmanager
    def db_opened(self, lock=False):
        if DISABLE_CACHING_PERSISTENCE:
            yield {}
            return

        from .resilience import retrying
        with ExitStack() as stack:
            with self.lock:
                try:
                    db = stack.enter_context(
                        retrying(3, acceptable=GDBMException, sleep=5)(shelve.open)(self.path))
                except Exception:
                    try:
                        os.unlink(self.path)
                    except FileNotFoundError:
                        pass
                    try:
                        db = stack.enter_context(shelve.open(self.path))
                    except Exception:
                        _logger.warning("Could not open PersistentCache: %s", self.path)
                        db = {}

            if lock:
                stack.enter_context(self.lock)
            yield db

    def set(self, key, value):
        with self.db_opened(lock=True) as db:
            if value == DELETED:
                del db[key]
            else:
                db[key] = (value, time.time())

    def get(self, key, default=NO_DEFAULT):
        try:
            with self.db_opened() as db:
                value, timestamp = db[key]
            if not self.expiration:
                pass
            elif (timestamp + self.expiration) <= time.time():
                raise KeyError()
            return value
        except KeyError:
            if default is NO_DEFAULT:
                raise
            else:
                return default

    def __call__(self, func=None, *, validator=None):
        if validator and not func:
            return partial(self.__call__, validator=validator)

        validator = validator and kwargs_resilient(validator)

        @wraps(func)
        def inner(*args, **kwargs):
            key_kwargs = {k: v for k, v in kwargs.items() if k not in self.ignored_keywords}
            key = str(_make_key(
                (func.__module__, func.__qualname__,) + args, key_kwargs,
                typed=False, kwd_mark=("_KWD_MARK_",)))

            try:
                value = self.get(key)
            except KeyError:
                pass
            else:
                if not validator:
                    return value
                validated_value = validator(value, args=args, kwargs=kwargs)
                if validated_value:
                    self.set(key, validated_value)
                    return validated_value
                self.set(key, DELETED)
                return inner(*args, **kwargs)
            ret = func(*args, **kwargs)
            self.set(key, ret)
            return ret
        inner.clear_cache = self.clear
        return inner

    def clear(self):
        with self.db_opened() as db:
            db.clear()


if sys.version_info < (3, 5):
    def _apply_defaults(bound_arguments):
        """
        Set default values for missing arguments. (from Python3.5)
        """
        from collections import OrderedDict
        from inspect import _empty, _VAR_POSITIONAL, _VAR_KEYWORD

        arguments = bound_arguments.arguments
        new_arguments = []
        for name, param in bound_arguments._signature.parameters.items():
            try:
                new_arguments.append((name, arguments[name]))
            except KeyError:
                if param.default is not _empty:
                    val = param.default
                elif param.kind is _VAR_POSITIONAL:
                    val = ()
                elif param.kind is _VAR_KEYWORD:
                    val = {}
                else:
                    # This BoundArguments was likely produced by
                    # Signature.bind_partial().
                    continue
                new_arguments.append((name, val))
        bound_arguments.arguments = OrderedDict(new_arguments)
else:
    def _apply_defaults(bound_arguments):
        bound_arguments.apply_defaults()


class _CachedException(tuple):
    pass


class _TimeCache(DecoratingDescriptor):

    def __init__(self, func, **kwargs):
        update_wrapper(self, func)  # this needs to be first to avoid overriding attributes we set
        super().__init__(func=func, cached=True)
        self.func = func
        self.kwargs = kwargs
        self.expiration = kwargs['expiration']
        self.get_ts_func = kwargs['get_ts_func']
        self.log_recalculation = kwargs['log_recalculation']
        self.cacheable_exceptions = kwargs['cacheable_exceptions']
        self.sig = inspect.signature(func)

        key_func = kwargs['key_func']
        if not key_func:
            def key_func(*args, **kwargs):
                return _make_key(args, kwargs, False)
        self._key_func = key_func

        self.NOT_FOUND = object()
        self.NOT_CACHED = self.NOT_FOUND, 0

        self.cache = {}
        self.main_lock = RLock()
        self.keyed_locks = defaultdict(RLock)

    def make_key(self, args, kwargs):
        bound = self.sig.bind(*args, **kwargs)
        _apply_defaults(bound)
        return kwargs_resilient(self._key_func)(**bound.arguments)

    def key_func(self, func):
        """
        Decorator for setting the key function for this cache.

        Example::

            >>> @timecache()
            ... def func(a, b):
            ...     ...

            >>> @func.key_func
            ... def func_key(*, a, b):
            ...     return (a, b, type(a))
        """
        self._key_func = func
        return func

    def __call__(self, *args, **kwargs):
        key = self.make_key(args, kwargs)

        with self.main_lock:
            key_lock = self.keyed_locks[key]
        with key_lock:
            result, ts = self.cache.get(key, self.NOT_CACHED)

            if self.expiration <= 0:
                pass  # nothing to fuss with, cache does not expire
            elif result is self.NOT_FOUND:
                pass  # cache is empty
            elif self.get_ts_func() - ts >= self.expiration:
                # cache expired
                result = self.NOT_FOUND
                del self.cache[key]

            if result is self.NOT_FOUND:
                if self.log_recalculation:
                    _logger.debug('time cache expired, calculating new value for %s', self.__name__)
                try:
                    result = self.func(*args, **kwargs)
                except self.cacheable_exceptions as exc:
                    result = _CachedException([exc])
                self.cache[key] = result, self.get_ts_func()

            if isinstance(result, _CachedException):
                raise result[0]
            else:
                return result

    def cache_clear(self):
        with self.main_lock:
            for key, lock in dict(self.keyed_locks).items():
                with lock:
                    self.cache.pop(key, None)

    def cache_pop(self, *args, **kwargs):
        key = self.make_key(args, kwargs)
        self.keyed_locks.pop(key, None)
        return self.cache.pop(key, None)

    def cache_push(self, *args, VALUE, TS=None, **kwargs):
        """
        Allows populating the cache without calling the wrapped function:

        Example::

            >>> @timecache()
            ... def func(a, b):
            ...     pass

            >>> func.cache_push(a=3, b=5, VALUE=8)

        :param VALUE: The value to push into the cache
        :param TS: The timestamp to associate with the value, for expiration.
        :type TS: float, optional
        """

        key = self.make_key(args, kwargs)
        with self.main_lock:
            key_lock = self.keyed_locks[key]
        with key_lock:
            self.cache[key] = VALUE, self.get_ts_func() if TS is None else TS

    def _decorate(self, method, instance, owner):
        return type(self)(method, **self.kwargs)


def timecache(expiration=0, get_ts_func=time.time, log_recalculation=False, key_func=None, cacheable_exceptions=()):
    """
    A thread-safe cache decorator with time expiration.

    :param expiration: if a positive number, set an expiration on the cache, defaults to 0
    :type expiration: number, optional
    :param get_ts_func: The function to be used in order to get the current time, defaults to time.time
    :type get_ts_func: callable, optional
    :param log_recalculation: Whether or not to log cache misses, defaults to False
    :type log_recalculation: bool, optional
    :param key_func: A function to use in order to create the item key
    :type key_func: callable, optional
    """

    def deco(func):
        return _TimeCache(
            func=func,
            expiration=expiration,
            get_ts_func=get_ts_func,
            log_recalculation=log_recalculation,
            key_func=key_func,
            cacheable_exceptions=cacheable_exceptions,
        )

    return deco


timecache.__doc__ = _TimeCache.__doc__


@parametrizeable_decorator
def locking_cache(func=None, log_recalculation=False, key_func=None, cacheable_exceptions=()):
    """
    A syntactic sugar for a locking cache without time expiration.

    :param log_recalculation: Whether or not to log cache misses, defaults to False
    :type log_recalculation: bool, optional
    :param key_func: A function to use in order to create the item key
    :type key_func: callable, optional
    """

    return timecache(log_recalculation=log_recalculation, key_func=key_func, cacheable_exceptions=cacheable_exceptions)(func)


@parametrizeable_decorator
def cached_property(function=None, locking=True, safe=True, cacheable_exceptions=()):
    """
    A property whose value is computed only once.

    :param function: Function to decorate, defaults to None
    :type function: function, optional
    :param locking: Lock cache access to make thread-safe, defaults to True
    :type locking: bool, optional
    :param safe: See `easypy.properties.safe_property`, defaults to True
    :type safe: bool, optional
    """
    return _cached_property(function=function, locking=locking, safe=safe, cacheable_exceptions=cacheable_exceptions)


class _cached_property(object):

    locks_lock = RLock()
    LOCKS_KEY = '__cached_properties_locks'

    def __init__(self, function, locking, safe, cacheable_exceptions):
        self._function = function
        self._locking = locking
        self._safe = safe
        self._attr_name = "_cached_%s" % self._function.__name__
        self._cacheable_exceptions = cacheable_exceptions

    def __set_name__(self, obj, name):
        self._attr_name = "_cached_%s" % name

    def __get__(self, obj, _=None):
        if obj is None:
            return self

        try:
            value = getattr(obj, self._attr_name)
        except AttributeError:
            pass
        else:
            if isinstance(value, _CachedException):
                raise value[0]
            else:
                return value

        with ExitStack() as stack:
            if self._locking:
                if not hasattr(obj, self.LOCKS_KEY):  # double synchronisation strategy to prevent more expensive lock entrance
                    with self.locks_lock:
                        if not hasattr(obj, self.LOCKS_KEY):
                            setattr(obj, self.LOCKS_KEY, defaultdict(RLock))
                stack.enter_context(getattr(obj, self.LOCKS_KEY)[self._attr_name])

                try:
                    # another thread has cached the value by the time we acquired the lock
                    return getattr(obj, self._attr_name)
                except AttributeError:
                    pass

            try:
                value = self._function(obj)
            except self._cacheable_exceptions as exc:
                value = _CachedException([exc])
            except AttributeError:
                if self._safe:
                    _, exc, tb = sys.exc_info()
                    raise RuntimeError("Attribute error within a property (%s)" % exc).with_traceback(tb)
                else:
                    raise

            setattr(obj, self._attr_name, value)

            if isinstance(value, _CachedException):
                raise value[0]
            else:
                return value

    def __set__(self, obj, value):
        setattr(obj, self._attr_name, value)

    def __delete__(self, obj):
        try:
            delattr(obj, self._attr_name)
        except AttributeError:
            pass

    def __call__(self, func):
        self._function = func
        return self
