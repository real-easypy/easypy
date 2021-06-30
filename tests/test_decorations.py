import pytest

from functools import wraps
from io import StringIO
from pydoc import doc

from easypy.decorations import lazy_decorator
from easypy.decorations import ensure_same_defaults, DefaultsMismatch
from easypy.decorations import kwargs_from
from easypy.misc import kwargs_resilient


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


def test_ensure_same_defaults():
    def foo(a=1, b=2, c=3):
        return a, b, c


    @ensure_same_defaults(foo)
    def bar(a=1, b=2, c=3):
        return a, b, c

    # Test we did not change the actual function
    assert foo() == bar()
    assert foo(4, 5, 6) == bar(4, 5, 6)


    with pytest.raises(DefaultsMismatch) as exc:
        @ensure_same_defaults(foo)
        def baz(a=1, b=3, c=2):
            pass
    assert exc.value.param_names == ['b', 'c']


def test_ensure_same_defaults_skipping_params_with_no_default():
    @ensure_same_defaults(lambda a=1, b=2: ...)
    def foo(a, b=2):
        pass

    @ensure_same_defaults(lambda a, b=2: ...)
    def foo(a=1, b=2):
        pass

    @ensure_same_defaults(lambda a, b=2: ...)
    def foo(a=1, c=3):
        pass

    with pytest.raises(DefaultsMismatch) as exc:
        @ensure_same_defaults(lambda a, b=2: ...)
        def foo(a, b=4):
            pass
    assert exc.value.param_names == ['b']


def test_ensure_same_defaults_ignore():
    @ensure_same_defaults(lambda a=1, b=2: ..., ignore=('b',))
    def foo(a=1, b=3):
        pass

    with pytest.raises(DefaultsMismatch) as exc:
        @ensure_same_defaults(lambda a=1, b=2: ..., ignore=('b',))
        def foo(a=2, b=3):
            pass
    assert exc.value.param_names == ['a']


def signature_line(thing):
    with StringIO() as capture:
        doc(thing, output=capture)
        return capture.getvalue().splitlines()[2]


def test_kwargs_from():
    import sys
    if sys.version_info < (3, 7):
        fix_annotations = lambda sign: sign.replace(': ', ':')
    else:
        fix_annotations = lambda sign: sign

    def foo(*, a, b: int, c=3):
        pass

    @kwargs_from(foo)
    def bar(a=1, **kwargs) -> bool:
        return False

    assert signature_line(bar) == fix_annotations('bar(a=1, *, b: int, c=3) -> bool')

    @kwargs_from(bar)
    def baz(c=4, **kwargs):
        pass

    assert signature_line(baz) == fix_annotations('baz(c=4, *, a=1, b: int)')


def test_kwargs_from_no_kwargs():
    with pytest.raises(TypeError):
        @kwargs_from(lambda *, x, y: ...)
        def foo(a):
            pass

    with pytest.raises(TypeError):
        @kwargs_from(lambda *, x, y: ...)
        def bar(x, y):
            pass

    @kwargs_from(lambda *args: ...)
    def baz(a, **kwargs):
        pass

    assert signature_line(baz) == 'baz(a)'


def test_kwargs_from_keep_kwargs():
    def foo(*, a, b, **foos):
        pass

    @kwargs_from(foo)
    def bar(x, **bars):
        pass

    assert signature_line(bar) == 'bar(x, *, a, b, **bars)'


def test_kwargs_from_multiple():
    @kwargs_from(lambda a, b: ..., lambda x, y, a: ...)
    def bar(**kwargs):
        pass

    assert signature_line(bar) == 'bar(*, a, b, x, y)'


def test_kwargs_from_exclude():
    @kwargs_from(lambda a, b, c, w: ..., exclude=['w'])
    def bar(**kwargs):
        pass

    assert signature_line(bar) == 'bar(*, a, b, c)'
