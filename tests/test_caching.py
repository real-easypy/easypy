import os
import time
from logging import getLogger
from uuid import uuid4

import pytest

from easypy.bunch import Bunch
from easypy.caching import timecache, PersistentCache
from easypy.units import DAY

_logger = getLogger(__name__)


def test_timecache():
    ts = 0
    data = Bunch(a=0, b=0)

    def get_ts():
        return ts

    @timecache(expiration=1, get_ts_func=get_ts)
    def inc(k):
        data[k] += 1

    assert data.a == data.b == 0
    inc('a')
    assert (data.a, data.b) == (1, 0)

    inc('a')
    assert (data.a, data.b) == (1, 0)

    ts += 1
    inc('a')
    assert (data.a, data.b) == (2, 0)

    inc('b')
    assert (data.a, data.b) == (2, 1)

    inc('b')
    assert (data.a, data.b) == (2, 1)

    ts += 1
    inc('b')
    assert (data.a, data.b) == (2, 2)

    inc.cache_clear()
    inc('a')
    assert (data.a, data.b) == (3, 2)
    inc('b')
    assert (data.a, data.b) == (3, 3)


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

    @ps(validator=lambda _: use_cache)
    def cached_func():
        nonlocal value_generated
        if value_generated:
            raise UnnecessaryFunctionCall()
        value_generated = True
        return True

    assert cached_func() is cached_func()
    assert value_generated

    # Testing validator
    use_cache = False
    with pytest.raises(UnnecessaryFunctionCall):
        cached_func()

    # Removing data
    ps.clear()
    assert ps.get(TEST_KEY, default=None) is None, "Database was not cleared properly"

    # Expiration
    ps = PersistentCache(persistent_cache_path, version=3, expiration=1 * DAY)
    ps.set(TEST_KEY, TEST_VALUE)
    ps.set("_PersistentCacheSignature", [3, time.time() - 2 * DAY])
    assert ps.get(TEST_KEY, None) is None, "Database was not cleaned up on expiration"
