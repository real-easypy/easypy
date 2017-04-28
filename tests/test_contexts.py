import pytest
from easypy.contexts import contextmanager


X = []


@contextmanager
def ctx():
    X.append(1)
    yield 2
    X.pop(0)


def test_simple():
    with ctx() as i:
        assert i == 2
        assert X == [1]
    assert not X


def test_function():
    @ctx()
    def foo():
        assert X == [1]
    foo()
    assert not X


def test_generator():
    @ctx()
    def foo():
        yield from range(5)

    for i in foo():
        assert X == [1]
    assert not X


def test_ctx():
    @ctx()
    @contextmanager
    def foo():
        yield

    with foo():
        assert X == [1]

    assert not X
