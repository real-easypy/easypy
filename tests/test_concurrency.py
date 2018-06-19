from mock import patch, call
import pytest
from time import sleep
import threading
import random
from contextlib import ExitStack

from easypy.threadtree import get_thread_stacks, ThreadContexts
from easypy.timing import TimeoutException
from easypy.concurrency import concurrent, MultiObject, MultiException
from easypy.concurrency import LoggedRLock, LockLeaseExpired
from easypy.concurrency import TagAlongThread
from easypy.concurrency import SynchronizedSingleton
from easypy.concurrency import LoggedCondition


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


def test_multiobject_1():
    m = MultiObject(range(10))

    def mul(a, b, *c):
        return a*b + sum(c)

    assert sum(m.call(mul, 2)) == 90
    assert sum(m.call(mul, b=10)) == 450
    assert sum(m.call(mul, 1, 1, 1)) == 65

    assert m.filter(None).L == [1, 2, 3, 4, 5, 6, 7, 8, 9]
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


def test_multiobject_concurrent_find_found():
    m = MultiObject(range(10))
    from time import sleep
    ret = m.concurrent_find(lambda n: sleep(n/10) or n)  # n==0 is not nonzero, so it's not eligible
    assert ret == 1


def test_multiobject_concurrent_find_not_found():
    m = MultiObject(range(10))
    ret = m.concurrent_find(lambda n: n < 0)
    assert ret is False

    m = MultiObject([0]*5)
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

    ret = m.zip_with(range(1, 5)).call(lambda a, b: a+b).L
    assert ret == [1, 3, 5, 7]


def test_multiobject_enumerate():
    m = MultiObject(range(5), log_ctx="abcd")

    def check(i, j):
        assert i == j + 1

    e = m.enumerate(1)
    assert e._log_ctx == list("abcd")
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


def test_logged_lock():
    lock = LoggedRLock("test", lease_expiration=1, log_interval=.2)

    step1 = threading.Event()
    step2 = threading.Event()

    def do_lock():
        with lock:
            step1.set()
            step2.wait()

    with concurrent(do_lock):
        # wait for thread to hold the lock
        step1.wait()

        # we'll mock the logger so we can ensure it logged
        with patch("easypy.concurrency._logger") as _logger:

            assert not lock.acquire(timeout=0.5)  # below the lease_expiration

            # the expiration mechanism should kick in
            with pytest.raises(LockLeaseExpired):
                lock.acquire()

        # let other thread finish
        step2.set()

    with lock:
        pass

    assert sum(c == call("%s - waiting...", lock) for c in _logger.debug.call_args_list) > 3


# this might be useful sometimes, but for now it didn't catch a bug
def disable_test_logged_lock_races():
    lease_expiration = 1
    lock = LoggedRLock("test", lease_expiration=lease_expiration, log_interval=.1)
    import logging

    def do_lock(idx):
        sleep(random.random())
        if lock.acquire(timeout=1, blocking=random.random() > 0.5):
            logging.info("%02d: acquired", idx)
            sleep(random.random() * lease_expiration * 0.9)
            lock.release()
            logging.info("%02d: released", idx)
        else:
            logging.info("%02d: timed out", idx)

    with ExitStack() as stack:
        for i in range(30):
            stack.enter_context(concurrent(do_lock, idx=i, loop=True, sleep=0))
        sleep(5)


def test_tag_along_thread():
    counter = 0

    def increment_counter():
        nonlocal counter
        counter += 1
        sleep(0.5)

    tag_along_thread = TagAlongThread(increment_counter, 'counter-incrementer')

    MultiObject(range(8)).call(lambda _: tag_along_thread())

    # The first call should get it's own iteration of the TagAlongThread. The
    # rest should wait for it to finish, and then all use the second iteration.
    # We don't have control over the timing inside the threads though, so we
    # allow fewer or more iterations - as long as an iteration did happen(at
    # least 1) and that invocations did stack together(less than 5 should do
    # it)
    assert 1 <= counter < 5


def test_sync_singleton():

    class S(metaclass=SynchronizedSingleton):
        def __init__(self):
            sleep(1)

    a, b = MultiObject(range(2)).call(lambda _: S())
    assert a is b


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


def test_logged_condition():
    cond = LoggedCondition('test', log_interval=.1)

    progress = 0
    executed = []

    def wait_for_progress_to(progress_to):
        cond.wait_for(lambda: progress_to <= progress, 'progress to %s', progress_to)
        executed.append(progress_to)

    with concurrent(wait_for_progress_to, 10), concurrent(wait_for_progress_to, 20), concurrent(wait_for_progress_to, 30):
        with patch("easypy.concurrency._logger") as _logger:
            sleep(0.3)

        assert any(c == call("%s - waiting for progress to %s", cond, 10) for c in _logger.debug.call_args_list)
        assert any(c == call("%s - waiting for progress to %s", cond, 20) for c in _logger.debug.call_args_list)
        assert any(c == call("%s - waiting for progress to %s", cond, 30) for c in _logger.debug.call_args_list)
        assert executed == []

        with patch("easypy.concurrency._logger") as _logger:
            with cond.notifying_all('setting progress to 10'):
                progress = 10
        assert [c for c in _logger.debug.call_args_list if 'performed' in c[0][0]] == [
            call("%s - performed: setting progress to 10", cond)]

        with patch("easypy.concurrency._logger") as _logger:
            sleep(0.3)

        assert not any(c == call("%s - waiting for progress to %s", cond, 10) for c in _logger.debug.call_args_list)
        assert any(c == call("%s - waiting for progress to %s", cond, 20) for c in _logger.debug.call_args_list)
        assert any(c == call("%s - waiting for progress to %s", cond, 30) for c in _logger.debug.call_args_list)
        assert executed == [10]

        with patch("easypy.concurrency._logger") as _logger:
            with cond.notifying_all('setting progress to 30'):
                progress = 30
        assert [c for c in _logger.debug.call_args_list if 'performed' in c[0][0]] == [
            call("%s - performed: setting progress to 30", cond)]

        with patch("easypy.concurrency._logger") as _logger:
            sleep(0.3)

        assert not any(c == call("%s - waiting for progress to %s", cond, 10) for c in _logger.debug.call_args_list)
        assert not any(c == call("%s - waiting for progress to %s", cond, 20) for c in _logger.debug.call_args_list)
        assert not any(c == call("%s - waiting for progress to %s", cond, 30) for c in _logger.debug.call_args_list)
        assert executed == [10, 20, 30] or executed == [10, 30, 20]

        with patch("easypy.concurrency._logger") as _logger:
            with pytest.raises(TimeoutException):
                cond.wait_for(lambda: False, 'the impossible', timeout=1)

        assert sum(c == call("%s - waiting for the impossible", cond) for c in _logger.debug.call_args_list) > 3


def test_logged_condition_exception():
    cond = LoggedCondition('test', log_interval=.2)

    should_throw = False

    class TestException(Exception):
        pass

    def waiter():
        if should_throw:
            raise TestException

    with pytest.raises(TestException):
        with concurrent(cond.wait_for, waiter, 'throw'):
            sleep(0.5)
            should_throw = True


def test_logged_condition_waited_for():
    cond = LoggedCondition('test', log_interval=15)
    progress = 0
    executed = []

    def wait_then_set_to(wait_to, set_to):
        nonlocal progress
        with cond.waited_for(lambda: progress == wait_to, 'progress to be %s', wait_to):
            executed.append('%s -> %s' % (progress, set_to))
            progress = set_to

    with concurrent(wait_then_set_to, 10, 20), concurrent(wait_then_set_to, 20, 30), concurrent(wait_then_set_to, 30, 40):
        with cond.notifying_all('setting progress to 10'):
            progress = 10

    assert executed == ['10 -> 20', '20 -> 30', '30 -> 40']
