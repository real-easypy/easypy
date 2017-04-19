import pytest
from easypy.collections import ListCollection, partial_dict
from easypy.collections import separate


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
