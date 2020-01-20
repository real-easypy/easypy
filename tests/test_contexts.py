import pytest
from easypy.contexts import contextmanager, breakable_section


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


def test_breakable_section():

    a = []
    with breakable_section() as Break1:
        with breakable_section() as Break2:
            with breakable_section() as Break3:
                raise Break2()
                a += [1]  # this will be skipped
            a += [2]  # this will be skipped
        a += [3]  # landing here
    a += [4]

    assert Break1 is not Break2
    assert Break2 is not Break3
    assert Break3 is not Break1
    assert a == [3, 4]
