import pytest

from functools import wraps

from easypy.decorations import deprecated_arguments, kwargs_resilient, lazy_decorator, singleton


def test_deprecated_arguments():
    @deprecated_arguments(foo='bar')
    def func(bar):
        return 'bar is %s' % (bar,)

    assert func(1) == func(foo=1) == func(bar=1) == 'bar is 1'

    with pytest.raises(TypeError):
        func(foo=1, bar=2)

    with pytest.raises(TypeError):
        func(1, foo=2)


def test_kwargs_resilient():
    @kwargs_resilient
    def foo(a, b):
        return [a, b]

    assert foo(1, b=2, c=3, d=4) == [1, 2]

    @kwargs_resilient
    def bar(a, b, **kwargs):
        return [a, b, kwargs]

    assert bar(1, b=2, c=3, d=4) == [1, 2, {'c': 3, 'd': 4}]

    @kwargs_resilient(negligible='d')
    def baz(a, b):
        return [a, b]

    # Should only be neglect `d` - not to `c`
    with pytest.raises(TypeError):
        baz(1, b=2, c=3, d=4)
    assert baz(1, b=2, d=4) == [1, 2]

    @kwargs_resilient(negligible=['b', 'd'])
    def qux(a, b, **kwargs):
        return [a, b, kwargs]

    # Should be passing b because it's in the function signature
    # Should be passing c because it's not in `negligible`
    # Should not be passing d because it's in `negligible` and not in the function signature
    assert qux(1, b=2, c=3, d=4) == [1, 2, {'c': 3}]


def test_lazy_decorator_lambda():
    def add_to_result(num):
        def inner(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs) + num

            wrapper.__name__ = '%s + %s' % (func.__name__, num)

            return wrapper
        return inner

    class Foo:
        def __init__(self, num):
            self.num = num

        @lazy_decorator(lambda self: add_to_result(num=self.num))
        def foo(self):
            """foo doc"""
            return 1

    foo = Foo(10)
    assert foo.foo() == 11

    assert Foo.foo.__name__ == 'foo'
    assert foo.foo.__name__ == 'foo + 10'

    assert Foo.foo.__doc__ == foo.foo.__doc__ == 'foo doc'


def test_lazy_decorator_attribute():
    class Foo:
        def add_to_result(self, func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs) + self.num

            wrapper.__name__ = '%s + %s' % (func.__name__, self.num)

            return wrapper

        @lazy_decorator('add_to_result')
        def foo(self):
            """foo doc"""
            return 1

    foo = Foo()

    with pytest.raises(AttributeError):
        # We did not set foo.num yet, so the decorator will fail trying to set the name
        foo.foo

    foo.num = 10
    assert foo.foo() == 11
    assert foo.foo.__name__ == 'foo + 10'
    assert Foo.foo.__doc__ == foo.foo.__doc__ == 'foo doc'

    foo.num = 20
    assert foo.foo() == 21
    assert foo.foo.__name__ == 'foo + 20'


def test_singleton():
    @singleton
    class foo:
        a = 1

        def __init__(self):
            self.b = 2

    assert foo.a == 1
    assert foo.b == 2
    assert type(foo).a == 1
    assert not hasattr(type(foo), 'b')
