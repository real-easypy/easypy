from contextlib import ExitStack, contextmanager
import sys
import os
import shelve
import time
import inspect
from collections import defaultdict
from functools import wraps, _make_key, partial, lru_cache
from threading import RLock
from logging import getLogger
from .resilience import retrying
from .decorations import kwargs_resilient, parametrizeable_decorator
from .collections import ilistify


_logger = getLogger(__name__)

try:
    from _gdbm import error as GDBMException
except:
    from _dbm import error as GDBMException


_disable_persistance = bool(os.getenv("bamboo_planKey", os.getenv("WEKA_job_key", None)))  # disable this feature when running from bamboo


class PersistentCache(object):
    """
    A memoizer that stores its cache persistantly using shelve.


    :param path: location of cache shelve file.

    :param version: Increment this to force-clear the cache.

    :param expiration: Expiration in seconds for the entire cache.
                    ``None`` to disable expiration

    Example::

        CACHE = PersistentCache("/tmp/cache", version=1, expiration=60)

        @CACHE
        def fib(n):
            a, b = 0, 1
            while a < n:
                a, b = b, a+b
            return a
    """

    DELETED = object()
    NONE = object()

    def __init__(self, path, version=0, expiration=60*60*4, ignored_keywords=None):
        self.path = path
        self.version = version
        self.expiration = expiration
        self.lock = RLock()
        self.ignored_keywords = set(ilistify(ignored_keywords)) if ignored_keywords else set()

    @contextmanager
    def db_opened(self, lock=False):
        if _disable_persistance:
            yield {}
            return

        with self.lock:
            try:
                db = retrying(3, acceptable=GDBMException, sleep=5)(shelve.open)(self.path)
            except:
                try:
                    os.unlink(self.path)
                except FileNotFoundError:
                    pass
                try:
                    db = shelve.open(self.path)
                except:
                    _logger.warning("Could not open PersistentCache: %s", self.path)
                    db = {}

        with ExitStack() as stack:
            timestamp = time.time()
            c_version, c_timestamp = db.get("_PersistentCacheSignature", [0, 0])
            if not ((c_version >= self.version) and (self.expiration is None or (c_timestamp + self.expiration) > timestamp)):
                with self.lock:
                    db.clear()
                    db["_PersistentCacheSignature"] = (self.version, timestamp)
            if lock:
                stack.enter_context(self.lock)
            yield db

        if type(db) is not dict:
            db.close()

    def set(self, key, value):
        with self.db_opened(lock=True) as db:
            if value == self.DELETED:
                del db[key]
            else:
                db[key] = value

    def get(self, key, default=NONE):
        try:
            with self.db_opened() as db:
                return db[key]
        except KeyError:
            if default is PersistentCache.NONE:
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
                self.set(key, self.DELETED)
                return inner(*args, **kwargs)
            ret = func(*args, **kwargs)
            self.set(key, ret)
            return ret
        inner.clear_cache = self.clear
        return inner

    def clear(self):
        with self.db_opened() as db:
            db.clear()


def locking_lru_cache(maxsize=128, typed=False):  # can't implement ignored_keywords because we use python's lru_cache...
    "An lru cache with a lock, to prevent concurrent invocations and allow reusing from cache"

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


def timecache(expiration=0, typed=False, get_ts_func=time.time, log_recalculation=False, ignored_keywords=None, key_func=None):
    """
    An lru cache with a lock, preventing concurrent invocations and allowing caching accross threads.

    If 'expiration'>0, set an expiration on the cache.
    'get_ts_func' can be used to provide an alternative timestamp for measuring expiration
    'ignored_keywords' can be used to ignore various arguments (positional or keyword) when generating the key for the cache
    'key_func' can be used to fully control how the key is generated
    """

    NOT_FOUND = object()
    NOT_CACHED = NOT_FOUND, 0

    if ignored_keywords:
        assert not key_func, "can't specify both `ignored_keywords` AND `key_func`"
        ignored_keywords = set(ilistify(ignored_keywords))

        def key_func(**kw):
            return tuple(v for k, v in sorted(kw.items()) if k not in ignored_keywords)

    def deco(func):
        cache = {}
        main_lock = RLock()
        keyed_locks = defaultdict(RLock)
        name = func.__name__
        sig = inspect.signature(func)

        @wraps(func)
        def inner(*args, **kwargs):
            if key_func:
                bound = sig.bind(*args, **kwargs)
                _apply_defaults(bound)
                key = kwargs_resilient(key_func)(**bound.arguments)
            else:
                key = _make_key(args, kwargs, typed=typed)

            with main_lock:
                key_lock = keyed_locks[key]
            with key_lock:
                result, ts = cache.get(key, NOT_CACHED)

                if expiration <= 0:
                    pass  # nothing to fuss with, cache does not expire
                elif result is NOT_FOUND:
                    pass  # cache is empty
                elif get_ts_func() - ts >= expiration:
                    # cache expired
                    result = NOT_FOUND
                    del cache[key]

                if result is NOT_FOUND:
                    if log_recalculation:
                        _logger.debug('time cache expired, calculating new value for %s', name)
                    result = func(*args, **kwargs)
                    cache[key] = result, get_ts_func()

                return result

        @wraps(func)
        def clear():
            with main_lock:
                for key, lock in dict(keyed_locks).items():
                    with lock:
                        cache.pop(key, None)

        inner.cache_clear = clear
        return inner

    return deco


@parametrizeable_decorator
def locking_cache(func=None, typed=False, log_recalculation=False, ignored_keywords=False):
    return timecache(typed=typed, log_recalculation=log_recalculation, ignored_keywords=ignored_keywords)(func)


class cached_property(object):
    """A property whose value is computed only once. """
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
