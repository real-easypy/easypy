from mock import patch, call
import pytest
from time import sleep
import threading
from easypy.threadtree import get_thread_stacks, ThreadContexts
from easypy.concurrency import concurrent, MultiObject, MultiException
from easypy.concurrency import LoggedRLock, LockLeaseExpired


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
