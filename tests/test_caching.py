import os
import time
from logging import getLogger
from uuid import uuid4
import random
import weakref
import gc

import pytest

from easypy.bunch import Bunch
from easypy.caching import timecache, PersistentCache, cached_property, locking_cache
from easypy.units import DAY
from easypy.resilience import resilient

_logger = getLogger(__name__)


def test_timecache():
    ts = 0
    data = Bunch(a=0, b=0)

    def get_ts():
        return ts

    @timecache(expiration=1, get_ts_func=get_ts, key_func=lambda k: k)
    def inc(k, x):
        x += 1
        data[k] += 1

    assert data.a == data.b == 0
    inc('a', random.random())
    assert (data.a, data.b) == (1, 0)

    inc('a', x=random.random())
    assert (data.a, data.b) == (1, 0)

    ts += 1
    inc('a', random.random())
    assert (data.a, data.b) == (2, 0)

    inc('b', x=random.random())
    assert (data.a, data.b) == (2, 1)

    inc('b', random.random())
    assert (data.a, data.b) == (2, 1)

    ts += 1
    inc('b', x=random.random())
    assert (data.a, data.b) == (2, 2)

    inc.cache_clear()
    inc('a', x=random.random())
    assert (data.a, data.b) == (3, 2)
    inc('b', x=random.random())
    assert (data.a, data.b) == (3, 3)

    inc.cache_clear()
    inc('a', x=random.random())
    inc('b', x=random.random())
    inc('a', x=random.random())
    inc('b', x=random.random())
    assert (data.a, data.b) == (4, 4)
    inc.cache_pop('a', x=random.random())
    inc('a', x=random.random())
    inc('b', x=random.random())


def test_timecache_method():
    ts = 0

    def get_ts():
        return ts

    class Foo:
        def __init__(self, prefix):
            self.prefix = prefix

        @timecache(expiration=1, get_ts_func=get_ts, key_func=lambda args: args)
        def foo(self, *args):
            return [self.prefix] + list(args)

    foo1 = Foo(1)
    foo2 = Foo(2)

    assert foo1.foo(1, 2, 3) == foo1.foo(1, 2, 3)
    assert foo1.foo(1, 2, 3) != foo1.foo(1, 2, 4)
    assert foo1.foo(1, 2, 3) != foo2.foo(1, 2, 3)

    foo1_1 = foo1.foo(1)
    foo1_2 = foo1.foo(2)
    foo2_1 = foo2.foo(1)
    foo2_2 = foo2.foo(2)

    assert foo1_1 == [1, 1]
    assert foo1_2 == [1, 2]
    assert foo2_1 == [2, 1]
    assert foo2_2 == [2, 2]

    assert foo1_1 is foo1.foo(1)
    assert foo1_2 is foo1.foo(2)
    assert foo2_1 is foo2.foo(1)
    assert foo2_2 is foo2.foo(2)

    assert foo1_1 is foo1.foo(1)
    assert foo1_2 is foo1.foo(2)
    assert foo2_1 is foo2.foo(1)
    assert foo2_2 is foo2.foo(2)

    foo1.foo.cache_clear()
    foo2.foo.cache_pop(1)

    assert foo1_1 is not foo1.foo(1)
    assert foo1_2 is not foo1.foo(2)
    assert foo2_1 is not foo2.foo(1)
    assert foo2_2 is foo2.foo(2)


def test_timecache_getattr():
    ts = 0

    def get_ts():
        return ts

    class Foo:
        def __init__(self):
            self.count = 0

        @timecache(expiration=1, get_ts_func=get_ts)
        def __getattr__(self, name):
            self.count += 1
            return [self.count, name]

    foo = Foo()

    assert foo.bar == [1, 'bar']
    assert foo.bar == [1, 'bar']
    assert foo.baz == [2, 'baz']

    ts += 1

    assert foo.baz == [3, 'baz']
    assert foo.bar == [4, 'bar']


@pytest.yield_fixture()
def persistent_cache_path():
    cache_path = '/tmp/test_pcache_%s' % uuid4()
    try:
        yield cache_path
    finally:
        try:
            os.unlink("%s.db" % cache_path)
        except:
            pass


def test_persistent_cache(persistent_cache_path):
    ps = PersistentCache(persistent_cache_path, version=1)
    TEST_KEY = "test_key"
    TEST_VALUE = "test_value"
    ps.set(TEST_KEY, TEST_VALUE)
    assert ps.get(TEST_KEY) == TEST_VALUE, "Value does not match set value"

    ps = PersistentCache(persistent_cache_path, version=1)
    assert ps.get(TEST_KEY) == TEST_VALUE, "Value does not match set value after reopen"

    ps = PersistentCache(persistent_cache_path, version=2)
    with pytest.raises(KeyError):  # Changed version should invalidate cache
        ps.get(TEST_KEY)

    # Default values
    assert ps.get(TEST_KEY, default=None) is None, "Wrong default value returnen(not None)"
    assert ps.get(TEST_KEY, default="1") is "1", "Wrong default value returned"

    # Cached func should be called only once
    value_generated = False
    use_cache = True

    class UnnecessaryFunctionCall(Exception):
        pass

    ps = PersistentCache(persistent_cache_path, version=2, ignored_keywords="x")

    @ps(validator=lambda _, **__: use_cache)
    def cached_func(x):
        nonlocal value_generated
        if value_generated:
            raise UnnecessaryFunctionCall()
        value_generated = True
        return True

    assert cached_func(x=random.random()) is cached_func(x=random.random())
    assert value_generated

    # Testing validator
    use_cache = False
    with pytest.raises(UnnecessaryFunctionCall):
        cached_func(x=random.random())

    # Removing data
    ps.clear()
    assert ps.get(TEST_KEY, default=None) is None, "Database was not cleared properly"

    # Expiration
    ps = PersistentCache(persistent_cache_path, version=3, expiration=.01)
    ps.set(TEST_KEY, TEST_VALUE)
    time.sleep(0.011)
    assert ps.get(TEST_KEY, None) is None, "Database was not cleaned up on expiration"


def test_locking_timecache():
    from easypy.concurrency import MultiObject

    # Cached func should be called only once
    value_generated = False

    class UnnecessaryFunctionCall(Exception):
        pass

    @timecache(ignored_keywords='x')
    def test(x):
        nonlocal value_generated
        if value_generated:
            raise UnnecessaryFunctionCall()
        value_generated = True
        return True

    MultiObject(range(10)).call(lambda x: test(x=x))


@pytest.mark.parametrize('cache_decorator', [cached_property, timecache()])
def test_caching_gc_leaks(cache_decorator):
    """
    Make sure that the cache does not prevent GC collection once the original objects die
    """

    class Leaked():
        pass

    class Foo:
        @cache_decorator
        def cached_method(self):
            return Leaked()

        def get(self):
            """Generalize property type and function type caches"""
            result = self.cached_method
            if callable(result):
                result = result()
            assert isinstance(result, Leaked), 'cache not used properly - got wrong value %s' % (result,)
            return result

    foo = Foo()
    leaked = weakref.ref(foo.get())

    gc.collect()
    assert leaked() == foo.get()

    del foo
    gc.collect()
    assert leaked() is None


def test_resilient_between_timecaches():
    class ExceptionLeakedThroughResilient(Exception):
        pass

    @timecache(1)
    @resilient(acceptable=ExceptionLeakedThroughResilient, default='default')
    @timecache(1)
    def foo():
        raise ExceptionLeakedThroughResilient()

    assert foo() == 'default'
