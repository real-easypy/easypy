import pytest
from easypy.exceptions import annotated_exceptions, TException
from easypy.bunch import Bunch


class T(TException):
    template = "The happened: {what}"


def test_pickle_texception():
    import pickle

    t1 = T(what="happened", a=1, b=Bunch(x=[1, 2, 3], y=range(5)))
    t2 = pickle.loads(pickle.dumps(t1))

    assert t1.render() == t2.render()
    assert t1._params == t2._params


def test_annotate_exceptions():
    class FooException(Exception):
        pass

    class BarException(TException):
        template = 'I am bar {a} {b}'

    # Doesn't throw if the inside doesn't throw
    with annotated_exceptions() as annotate:
        annotate(baz='qux')

    try:
        with annotated_exceptions() as annotate:
            annotate(baz='qux')
            raise FooException('this is a foo exception')
    except BarException:
        assert False, 'Caught bar when we were throwing foo'
    except FooException as e:
        assert isinstance(e, FooException)
        assert issubclass(type(e), FooException)

        assert 'this is a foo exception' in str(e)
        assert e.baz == 'qux', 'should have been annotated'
        assert 'baz = qux' in str(e)
    else:
        assert False, 'Exception got swallowed'

    with pytest.raises(BarException) as exc:
        with annotated_exceptions() as annotate:
            annotate(baz='qux')
            raise BarException(a='1', b='2')
    assert type(exc.value) is BarException, 'subclass of PException got wrapped'


def test_annotate_builtin_exceptions():
    with pytest.raises(ZeroDivisionError) as exc:
        with annotated_exceptions() as annotate:
            numerator = 10
            annotate(numerator=numerator)
            numerator / 0

    assert 'division by zero' in str(exc.value)
    assert exc.value.numerator == 10
    assert 'numerator = 10' in str(exc.value)
