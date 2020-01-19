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


def test_aliasing_infinite_recursion_exception():
    @aliases('bar', static=False)
    class Foo:
        def __init__(self):
            self.bar = Bar(self)

        def __repr__(self):
            return 'Foo()'

    class Bar:
        def __init__(self, foo):
            self.foo = foo

        def baz(self):
            return self.foo.baz()

    with pytest.raises(getattr(__builtins__, 'RecursionError', RuntimeError)) as e:
        Foo().baz()
    assert str(e.value) == "Infinite recursion trying to access 'baz' on Foo() (via Foo.bar.baz)"
