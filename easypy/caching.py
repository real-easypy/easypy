import os
import sys
import time
import shelve
import inspect
from threading import RLock
from logging import getLogger
from collections import defaultdict
from contextlib import ExitStack, contextmanager
from functools import wraps, _make_key, partial, lru_cache, update_wrapper

from .decorations import parametrizeable_decorator, DecoratingDescriptor
from .collections import ilistify
from .misc import kwargs_resilient
from .deprecation import deprecated
from .units import HOUR
from .tokens import DELETED, NO_DEFAULT
from .humanize import yesno_to_bool

_logger = getLogger(__name__)

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


@deprecated(message="Please use easypy.caching.locking_cache")
def locking_lru_cache(maxsize=128, typed=False):
    """
    DEPRECATED!

    An lru cache decorator with a lock, to prevent concurrent invocations and allow reusing from cache.

    :param maxsize: LRU cache maximum size, defaults to 128
    :type maxsize: number, optional
    :param typed: If typed is set to true, function arguments of different types will be cached separately. defaults to False.
    :type typed: bool, optional
    """

    def deco(func):
        caching_func = lru_cache(maxsize, typed)(func)
        func._main_lock = RLock()
        func._keyed_locks = defaultdict(RLock)

        @wraps(func)
        def inner(*args, **kwargs):
            key = _make_key(args, kwargs, typed=typed)
            with func._main_lock:
                key_lock = func._keyed_locks[key]
            with key_lock:
                return caching_func(*args, **kwargs)

        @wraps(caching_func.cache_clear)
        def clear():
            with func._main_lock:
                return caching_func.cache_clear()

        inner.cache_clear = clear
        return inner

    return deco


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


class _TimeCache(DecoratingDescriptor):
    def __init__(self, func, **kwargs):
        update_wrapper(self, func)  # this needs to be first to avoid overriding attributes we set
        super().__init__(func=func, cached=True)
        self.func = func
        self.kwargs = kwargs
        self.expiration = kwargs['expiration']
        self.typed = kwargs['typed']
        self.get_ts_func = kwargs['get_ts_func']
        self.log_recalculation = kwargs['log_recalculation']
        self.ignored_keywords = kwargs['ignored_keywords']

        if self.ignored_keywords:
            assert not kwargs['key_func'], "can't specify both `ignored_keywords` AND `key_func`"
            self.ignored_keywords = set(ilistify(self.ignored_keywords))

            def key_func(**kw):
                return tuple(v for k, v in sorted(kw.items()) if k not in self.ignored_keywords)
        else:
            key_func = kwargs['key_func']

        self.NOT_FOUND = object()
        self.NOT_CACHED = self.NOT_FOUND, 0

        self.cache = {}
        self.main_lock = RLock()
        self.keyed_locks = defaultdict(RLock)

        if key_func:
            sig = inspect.signature(func)

            def make_key(args, kwargs):
                bound = sig.bind(*args, **kwargs)
                _apply_defaults(bound)
                return kwargs_resilient(key_func)(**bound.arguments)
        else:
            def make_key(args, kwargs):
                return _make_key(args, kwargs, typed=self.typed)

        self.make_key = make_key

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
                result = self.func(*args, **kwargs)
                self.cache[key] = result, self.get_ts_func()

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

    def _decorate(self, method, instance, owner):
        return type(self)(method, **self.kwargs)


def timecache(expiration=0, typed=False, get_ts_func=time.time, log_recalculation=False, ignored_keywords=None, key_func=None):
    """
    A thread-safe cache decorator with time expiration.

    :param expiration: if a positive number, set an expiration on the cache, defaults to 0
    :type expiration: number, optional
    :param typed: If typed is set to true, function arguments of different types will be cached separately, defaults to False
    :type typed: bool, optional
    :param get_ts_func: The function to be used in order to get the current time, defaults to time.time
    :type get_ts_func: callable, optional
    :param log_recalculation: Whether or not to log cache misses, defaults to False
    :type log_recalculation: bool, optional
    :param ignored_keywords: Arguments to ignore when caculating item key, defaults to None
    :type ignored_keywords: iterable, optional
    :param key_func: The function to use in order to create the item key, defaults to functools._make_key
    :type key_func: callable, optional
    """

    def deco(func):
        return _TimeCache(
            func=func,
            expiration=expiration,
            typed=typed,
            get_ts_func=get_ts_func,
            log_recalculation=log_recalculation,
            ignored_keywords=ignored_keywords,
            key_func=key_func)

    return deco


timecache.__doc__ = _TimeCache.__doc__


@parametrizeable_decorator
def locking_cache(func=None, typed=False, log_recalculation=False, ignored_keywords=False):
    """
    A syntactic sugar for a locking cache without time expiration.

    :param typed: If typed is set to true, function arguments of different types will be cached separately, defaults to False
    :type typed: bool, optional
    :param log_recalculation: Whether or not to log cache misses, defaults to False
    :type log_recalculation: bool, optional
    :param ignored_keywords: Arguments to ignore when caculating item key, defaults to None
    :type ignored_keywords: iterable, optional
    """

    return timecache(typed=typed, log_recalculation=log_recalculation, ignored_keywords=ignored_keywords)(func)


class cached_property(object):
    """
    A property whose value is computed only once.

    :param function: Function to decorate, defaults to None
    :type function: function, optional
    :param locking: Lock cache access to make thread-safe, defaults to True
    :type locking: bool, optional
    :param safe: See `easypy.properties.safe_property`, defaults to True
    :type safe: bool, optional
    """

    locks_lock = RLock()
    LOCKS_KEY = '__cached_properties_locks'

    def __init__(self, function=None, locking=True, safe=True):
        self._function = function
        self._locking = locking
        self._safe = safe

    def __get__(self, obj, _=None):
        if obj is None:
            return self
        func_name = self._function.__name__
        with ExitStack() as stack:
            if self._locking:
                if not hasattr(obj, self.LOCKS_KEY):  # double synchronisation strategy to prevent more expensive lock entrance
                    with self.locks_lock:
                        if not hasattr(obj, self.LOCKS_KEY):
                            setattr(obj, self.LOCKS_KEY, defaultdict(RLock))
                stack.enter_context(getattr(obj, self.LOCKS_KEY)[func_name])

                try:
                    # another thread has cached the value by the time we acquired the lock
                    return getattr(obj, "_cached_%s" % func_name)
                except AttributeError:
                    pass

            try:
                value = self._function(obj)
            except AttributeError:
                if self._safe:
                    _, exc, tb = sys.exc_info()
                    raise RuntimeError("Attribute error within a property (%s)" % exc).with_traceback(tb)
                else:
                    raise

            setattr(obj, "_cached_%s" % func_name, value)
            setattr(obj, func_name, value)
            return value

    def __call__(self, func):
        self._function = func
        return self
