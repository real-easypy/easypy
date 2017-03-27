import pytest
from easypy.caching import timecache
from easypy.bunch import Bunch


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
