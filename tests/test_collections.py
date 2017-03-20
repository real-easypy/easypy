from easypy.collections import ListCollection, partial_dict


def test_collection_filter():
    l = ListCollection("abcdef")
    assert l.filtered(lambda c: c == 'a').sample(1) == ['a']


def test_partial_dict():
    assert partial_dict({'a': 1, 'b': 2, 'c': 3}, ['a', 'b']) == {'a': 1, 'b': 2}
