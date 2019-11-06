from unittest.mock import patch
import pytest
from time import sleep

from easypy.threadtree import get_thread_stacks, ThreadContexts
from easypy.concurrency import concurrent, MultiObject, MultiException


@pytest.yield_fixture(params=[True, False], ids=['concurrent', 'nonconcurrent'])
def concurrency_enabled_and_disabled(request):
    if request.param:  # concurrency enabled
        yield
    else:  # concurrency disabled
        from easypy.concurrency import disable, enable
        try:
            disable()
            yield
        finally:
            enable()


def test_thread_stacks():
    with concurrent(sleep, .1, threadname='sleep'):
        print(get_thread_stacks().render())


def test_thread_contexts_counters():
    TC = ThreadContexts(counters=('i', 'j'))
    assert TC.i == TC.j == 0

    with TC(i=1):
        def check1():
            assert TC.i == 1
            assert TC.j == 0

            with TC(i=1, j=1):
                def check2():
                    assert TC.i == 2
                    assert TC.j == 1
                with concurrent(check2):
                    pass

        with concurrent(check1):
            pass


def test_thread_contexts_counters_multiobject():
    TC = ThreadContexts(counters=('i',))
    assert TC.i == 0

    print("---")

    @TC(i=True)
    def test(n):
        print(n, TC._context_data)
        sleep(.1)
        return TC.i

    test(0)
    ret = MultiObject(range(10)).call(test)
    assert set(ret) == {1}


def test_thread_context_stacks():
    TC = ThreadContexts(stacks=('i', 'j'))
    assert TC.i == TC.j == []

    with TC(i='a'):
        def check1():
            assert TC.i == ['a']
            assert TC.j == []

            with TC(i='i', j='j'):
                def check2():
                    assert TC.i == ['a', 'i']
                    assert TC.j == ['j']
                with concurrent(check2):
                    pass

        with concurrent(check1):
            pass


def test_multiobject_0():
    x = MultiObject([]).foo()
    assert len(x) == 0
    assert x.__class__.CONCESTOR is object


def test_multiobject_1():
    m = MultiObject(range(10))

    def mul(a, b, *c):
        return a * b + sum(c)

    assert sum(m.call(mul, 2)) == 90
    assert sum(m.call(mul, b=10)) == 450
    assert sum(m.call(mul, 1, 1, 1)) == 65

    assert m.filter(None).T == (1, 2, 3, 4, 5, 6, 7, 8, 9)
    assert sum(m.denominator) == 10
    with pytest.raises(MultiException) as info:
        m.call(lambda i: 1 / (i % 2))

    assert info.value.count == 5
    assert info.value.common_type == ZeroDivisionError
    assert not info.value.complete


def test_multiobject_exceptions():

    assert MultiException[ValueError] is MultiException[ValueError]
    assert issubclass(MultiException[UnicodeDecodeError], MultiException[UnicodeError])
    assert issubclass(MultiException[UnicodeDecodeError], MultiException[ValueError])

    with pytest.raises(AssertionError):
        MultiException[0]

    with pytest.raises(MultiException):
        MultiObject(range(5)).call(lambda n: 1 / n)

    with pytest.raises(MultiException[Exception]):
        MultiObject(range(5)).call(lambda n: 1 / n)

    with pytest.raises(MultiException[ZeroDivisionError]):
        MultiObject(range(5)).call(lambda n: 1 / n)

    try:
        MultiObject(range(5)).call(lambda n: 1 / n)
    except MultiException[ValueError] as exc:
        assert False
    except MultiException[ZeroDivisionError] as exc:
        assert len(exc.actual) == 1
        assert isinstance(exc.one, ZeroDivisionError)
    else:
        assert False

    with pytest.raises(MultiException[ArithmeticError]):
        try:
            MultiObject(range(5)).call(lambda n: 1 / n)
        except ZeroDivisionError:
            assert False  # shouldn't be here
        except MultiException[ValueError]:
            assert False  # shouldn't be here


class ExceptionForPicklingTest(ArithmeticError):
    pass


def test_multiexception_pickling():
    import pickle
    import multiprocessing

    def throw(n):
        if not n:
            raise ExceptionForPicklingTest(n)

    def fail_and_dump(queue):
        try:
            MultiObject(range(5)).call(throw)
        except MultiException[ArithmeticError] as exc:
            p = pickle.dumps(exc)
            queue.put_nowait(p)

    queue = multiprocessing.Queue(1)
    process = multiprocessing.Process(target=fail_and_dump, args=(queue,))
    process.start()
    process.join()
    p = queue.get_nowait()

    exc = pickle.loads(p)
    assert isinstance(exc, MultiException[ExceptionForPicklingTest])
    assert exc.common_type is ExceptionForPicklingTest
    assert exc.exceptions[0].args == (0,)
    assert exc.exceptions[1:] == [None] * 4


def test_multiobject_concurrent_find_found():
    m = MultiObject(range(10))
    from time import sleep
    ret = m.concurrent_find(lambda n: sleep(n / 10) or n)  # n==0 is not nonzero, so it's not eligible
    assert ret == 1


def test_multiobject_concurrent_find_not_found():
    m = MultiObject(range(10))
    ret = m.concurrent_find(lambda n: n < 0)
    assert ret is False

    m = MultiObject([0] * 5)
    ret = m.concurrent_find(lambda n: n)
    assert ret is 0


def test_multiobject_concurrent_find_proper_shutdown():
    executed = []
    m = MultiObject(range(10), workers=1)
    ret = m.concurrent_find(lambda n: [print(n) or executed.append(n) or sleep(.01)])
    assert ret
    sleep(1)  # wait for potential stragglers
    assert max(executed) <= 2


def test_multiobject_zip_with():
    m = MultiObject(range(4))

    with pytest.raises(AssertionError):
        m.zip_with(range(3), range(5))  # too few objects

    m.zip_with(range(5), range(6))  # too many objects

    ret = m.zip_with(range(1, 5)).call(lambda a, b: a + b).T
    assert ret == (1, 3, 5, 7)


def test_multiobject_enumerate():
    m = MultiObject(range(5), log_ctx="abcd")

    def check(i, j):
        assert i == j + 1

    e = m.enumerate(1)
    assert e._log_ctx == tuple("abcd")
    e.call(check)


def test_multiobject_logging():
    m = MultiObject(range(4), log_ctx="abcd", initial_log_interval=0.1)

    def check(i):
        sleep(.2)

    # we'll mock the logger so we can ensure it logged
    with patch("easypy.concurrency._logger") as _logger:
        m.call(check)

    args_list = (c[0] for c in _logger.info.call_args_list)
    for args in args_list:
        assert "test_multiobject_logging.<locals>.check" == args[2]
        assert "easypy/tests/test_concurrency.py" in args[4]


def test_multiobject_types():
    assert isinstance(MultiObject(range(5)), MultiObject[int])
    assert not isinstance(MultiObject(range(5)), MultiObject[str])

    class A(): ...
    class B(A): ...

    assert issubclass(MultiObject[A], MultiObject)
    assert not issubclass(MultiObject[A], A)
    assert issubclass(MultiObject[B], MultiObject[A])
    assert not issubclass(MultiObject[A], MultiObject[B])

    assert isinstance(MultiObject([B()]), MultiObject[A])
    assert not isinstance(MultiObject([A()]), MultiObject[B])
    assert isinstance(MultiObject[A]([B()]), MultiObject[A])
    assert isinstance(MultiObject[A]([B()]), MultiObject[B])
    assert isinstance(MultiObject[int](range(5)), MultiObject[int])

    with pytest.raises(TypeError):
        assert MultiObject[str](range(5))

    assert isinstance(MultiObject[str]("123").call(int), MultiObject[int])


def test_multiobject_namedtuples():
    from collections import namedtuple

    class Something(namedtuple("Something", "a b")):
        pass

    def ensure_not_expanded(something):
        # This will probably fail before these asserts
        assert hasattr(something, 'a')
        assert hasattr(something, 'b')

    objects = [Something(1, 2), Something(2, 3), Something(3, 4)]
    MultiObject(objects).call(ensure_not_expanded)


@pytest.mark.usefixtures('concurrency_enabled_and_disabled')
def test_multiexception_api():
    with pytest.raises(MultiException) as exc:
        MultiObject([0, 5]).call(lambda i: 10 // i)

    failed, sucsessful = exc.value.futures

    assert failed.done()
    with pytest.raises(ZeroDivisionError):
        failed.result()
    assert isinstance(failed.exception(), ZeroDivisionError)

    assert sucsessful.done()
    assert sucsessful.result() == 2
    assert sucsessful.exception() is None


def test_multiexception_types():

    class OK(Exception):
        pass

    class BAD(object):
        pass

    class OKBAD(OK, BAD):
        pass

    with pytest.raises(AssertionError):
        MultiException[BAD]

    def raise_it(typ):
        raise typ()

    with pytest.raises(MultiException[OK]):
        MultiObject([OK]).call(raise_it)

    with pytest.raises(MultiException[OKBAD]):
        MultiObject([OKBAD]).call(raise_it)

    with pytest.raises(MultiException[OK]):
        MultiObject([OKBAD]).call(raise_it)


@pytest.mark.parametrize('throw', [False, True])
def test_concurrent_done_status(throw):
    from threading import Event

    continue_func = Event()

    def func():
        continue_func.wait()
        if throw:
            raise Exception()

    with concurrent(func, throw=False) as c:
        assert not c.done()
        continue_func.set()
        sleep(0.1)
        assert c.done()
    assert c.done()
