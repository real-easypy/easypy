import pytest
from easypy.bunch import Bunch, bunchify, MissingRequiredKeys, CannotDeleteRequiredKey, KeyNotAllowed


def test_bunch_recursion():
    x = Bunch(a='a', b="b", d=Bunch(x="axe", y="why"))
    x.d.x = x
    x.d.y = x.b
    print(x)


def test_bunchify():
    x = bunchify(dict(a=[dict(b=5), 9, (1, 2)], c=8))
    assert x.a[0].b == 5
    assert x.a[1] == 9
    assert isinstance(x.a[2], tuple)
    assert x.c == 8


def test_required_keys():

    class RB(Bunch):
        KEYS = frozenset("a b c".split())

    with pytest.raises(KeyNotAllowed):
        x = RB(a='a', b='d', c='c', d='d')

    with pytest.raises(KeyNotAllowed):
        x = RB.fromkeys("abcd", True)

    x = RB(a='a', b='b', c='c')
    assert 'a' in x
    assert x['b']
    assert x['c']

    x = RB.fromkeys("abc", True)
    assert 'a' in x
    assert x['b']
    assert x['c']

    with pytest.raises(CannotDeleteRequiredKey):
        x.pop("a")

    with pytest.raises(MissingRequiredKeys):
        x = RB()

    with pytest.raises(MissingRequiredKeys):
        x = RB(a='a')

    with pytest.raises(MissingRequiredKeys):
        x = RB(a='a', d=0)

    with pytest.raises(MissingRequiredKeys):
        x = RB.fromkeys("ab")

    with pytest.raises(MissingRequiredKeys):
        x = RB.fromkeys("abd")

    x = RB.from_yaml("{a: a, b: b, c: c}")
    with pytest.raises(MissingRequiredKeys):
        x = RB.from_yaml("{a: a, b: b, d: d}")
