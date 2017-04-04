import pytest
from easypy.collections import ListCollection, partial_dict


def test_collection_filter():
    l = ListCollection("abcdef")
    assert l.filtered(lambda c: c == 'a').sample(1) == ['a']


def test_partial_dict():
    assert partial_dict({'a': 1, 'b': 2, 'c': 3}, ['a', 'b']) == {'a': 1, 'b': 2}


def test_collection_sample():
    l = ListCollection("abcdef")
    assert len(l.sample(2.0)) == 2

    with pytest.raises(AssertionError):
        l.sample(1.5)
