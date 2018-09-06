from easypy.bunch import Bunch, bunchify


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
    assert x.pop("c") == 8
