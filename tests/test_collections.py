import pytest
from easypy.collections import separate
from easypy.collections import ListCollection, partial_dict, UNIQUE, ObjectNotFound
from easypy.collections import ContextCollection
from easypy.bunch import Bunch
from collections import Counter
from contextlib import ExitStack


class O(Bunch):
    def __repr__(self):
        return "%(name)s:%(id)s:%(v)s" % self

L = ListCollection(O(name=n, id=i, v=v) for n, i, v in zip("aabcdddeff", "1234567890", "xxxyyyxxyz") for _ in range(100))


def test_collection_filter():
    l = ListCollection("abcdef")
    assert l.filtered(lambda c: c == 'a').sample(1) == ['a']


def test_partial_dict():
    assert partial_dict({'a': 1, 'b': 2, 'c': 3}, ['a', 'b']) == {'a': 1, 'b': 2}


def test_separate1():
    a, b = separate(range(5), key=lambda n: n < 3)
    assert a == [0, 1, 2]
    assert b == [3, 4]


def test_separate2():
    a, b = separate(range(5))
    assert a == [1, 2, 3, 4]
    assert b == [0]


def test_collection_sample():
    l = ListCollection("abcdef")
    assert len(l.sample(2.0)) == 2

    with pytest.raises(AssertionError):
        l.sample(1.5)


def test_collection_select():
    assert len(L.select(name='a', id='1')) == 100


def test_collection_select_no_unique():
    with pytest.raises(AssertionError):
        L.select(name=UNIQUE)


def test_collection_sample_too_much():
    len(L.select(name='a', id='2').sample(100)) == 100
    with pytest.raises(ObjectNotFound):
        L.select(name='a', id='2').sample(101)


def test_collection_sample_unique0():
    assert not L.sample(0, name=UNIQUE)


def test_collection_sample_unique1():
    s = L.sample(3, name=UNIQUE)
    assert len({b.name for b in s}) == 3


def test_collection_sample_unique2():
    x, = L.sample(1, name=UNIQUE, id='1')
    assert x.id == '1'


def test_collection_sample_unique3():
    s = L.sample(6, name=UNIQUE, id=UNIQUE)
    assert len({(b.name, b.id) for b in s}) == 6


def test_collection_sample_unique4():
    with pytest.raises(ObjectNotFound):
        L.sample(7, name=UNIQUE)  # too many


def test_collection_sample_unique5():
    s = L.sample(3, name=UNIQUE, id=UNIQUE, v=UNIQUE)
    assert len({(b.name, b.id) for b in s}) == 3


def test_collection_sample_unique_diverse():
    x = Counter(repr(x) for _ in range(100) for x in L.sample(1, name=UNIQUE))
    assert len(x) == 10


def test_collections_slicing():
    L = ListCollection("abcdef")
    assert L[0] == 'a'
    assert L[-1] == 'f'
    assert L[:2] == list('ab')
    assert L[-2:] == list('ef')
    assert L[::2] == list('ace')
    assert L[::-2] == list('fdb')


def test_context_collection():
    cc = ContextCollection()

    assert list(cc) == []
    with cc.added(10):
        assert list(cc) == [10]
        with cc.added(20):
            assert list(cc) == [10, 20]
        assert list(cc) == [10]
    assert list(cc) == []

    with ExitStack() as stack:
        with cc.added(30):
            assert list(cc) == [30]

            stack.enter_context(cc.added(40))
            assert list(cc) == [30, 40]

        assert list(cc) == [40]

        with cc.added(50):
            assert list(cc) == [40, 50]
        assert list(cc) == [40]
    assert list(cc) == []
