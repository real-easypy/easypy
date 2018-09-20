import pytest

from functools import wraps

from easypy.decorations import deprecated_arguments, kwargs_resilient, lazy_decorator, singleton_contextmanager


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


def test_lazy_decorator_with_timecache():
    from easypy.caching import timecache

    class Foo:
        def __init__(self):
            self.ts = 0
            self._counter = 0

        @property
        def timecache(self):
            return timecache(expiration=1, get_ts_func=lambda: self.ts)

        @lazy_decorator('timecache', cached=True)
        def inc(self):
            self._counter += 1
            return self._counter

        @lazy_decorator(lambda self: lambda method: method())
        @lazy_decorator('timecache', cached=True)
        def counter(self):
            return self._counter

    foo1 = Foo()
    foo2 = Foo()

    assert [foo1.inc(), foo2.inc()] == [1, 1]
    assert [foo1.inc(), foo2.inc()] == [1, 1]
    assert [foo1.counter, foo2.counter] == [1, 1]

    foo1.ts += 1
    assert [foo1.counter, foo2.counter] == [1, 1]
    assert [foo1.inc(), foo2.inc()] == [2, 1]
    assert [foo1.counter, foo2.counter] == [1, 1]
    foo2.ts += 1
    assert [foo1.inc(), foo2.inc()] == [2, 2]
    assert [foo1.counter, foo2.counter] == [1, 2]  # foo1 was not updated since last sync - only foo2


def test_singleton_contextmanager():

    data = []

    @singleton_contextmanager
    def foo(a):
        data.append(a)
        yield a
        data.append(-a)

    with foo(1):
        assert data == [1]

        with foo(2):
            assert data == [1, 2]

            with foo(2):
                assert data == [1, 2]

                with foo(1):
                    assert data == [1, 2]

            assert data == [1, 2]
        assert data == [1, 2, -2]
    assert data == [1, 2, -2, -1]

    data.clear()

    with foo(1):
        assert data == [1]
        with foo(2):
            assert data == [1, 2]
            with foo(2):
                assert data == [1, 2]
            assert data == [1, 2]
        assert data == [1, 2, -2]
    assert data == [1, 2, -2, -1]


def test_singleton_contextmanager_method():

    class Foo(object):

        def __init__(self):
            self.data = []

        @singleton_contextmanager
        def foo(self, a):
            self.data.append(a)
            yield a
            self.data.append(-a)

    f = Foo()
    g = Foo()

    with f.foo(1), g.foo(5):
        assert f.data == [1]
        assert g.data == [5]

        with f.foo(2), g.foo(6):
            assert f.data == [1, 2]
            assert g.data == [5, 6]

            with f.foo(2), g.foo(6):
                assert f.data == [1, 2]
                assert g.data == [5, 6]

                with f.foo(1), g.foo(5):
                    assert f.data == [1, 2]
                    assert g.data == [5, 6]

            assert f.data == [1, 2]
            assert g.data == [5, 6]
        assert f.data == [1, 2, -2]
        assert g.data == [5, 6, -6]
    assert f.data == [1, 2, -2, -1]
    assert g.data == [5, 6, -6, -5]

    f.data.clear()
    g.data.clear()

    with f.foo(1), g.foo(5):
        assert f.data == [1]
        assert g.data == [5]

        with f.foo(2), g.foo(6):
            assert f.data == [1, 2]
            assert g.data == [5, 6]

            with f.foo(2), g.foo(6):
                assert f.data == [1, 2]
                assert g.data == [5, 6]

                with f.foo(1), g.foo(5):
                    assert f.data == [1, 2]
                    assert g.data == [5, 6]

            assert f.data == [1, 2]
            assert g.data == [5, 6]
        assert f.data == [1, 2, -2]
        assert g.data == [5, 6, -6]
    assert f.data == [1, 2, -2, -1]
    assert g.data == [5, 6, -6, -5]


def test_singleton_contextmanager_method_does_not_keep_object_alive_after_done():
    import weakref
    import gc

    class Foo:
        @singleton_contextmanager
        def foo(self):
            yield

    foo = Foo()
    weak_foo = weakref.ref(foo)
    with foo.foo():
        del foo
        gc.collect()
        assert weak_foo() is not None, 'Object collected but contextmanager is active'
    gc.collect()
    assert weak_foo() is None, 'Object not collected but contextmanager has exited'
