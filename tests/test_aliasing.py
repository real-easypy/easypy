import pytest
from easypy.aliasing import aliases


def test_aliasing_static():

    @aliases("this")
    class Foo():
        this = dict(a=1)

    f = Foo()
    assert f.get("a") == 1


def test_aliasing_dynamic():

    class Foo():
        def __init__(self):
            self.this = dict(a=1)

    with pytest.raises(AssertionError):
        Foo = aliases("this")(Foo)

    Foo = aliases("this", static=False)(Foo)

    f = Foo()
    assert f.get("a") == 1


def test_aliasing_inherit():

    class Foo(int):
        def __init__(self, x):
            self.this = dict(a=1)

    with pytest.raises(AssertionError):
        Foo = aliases("this")(Foo)

    Foo = aliases("this", static=False)(Foo)

    f = Foo("5")
    assert f.get("a") == 1
    assert f == 5
