from unittest.mock import patch, call
import pytest
from time import sleep
import threading
from contextlib import contextmanager, ExitStack
import random
import re

from easypy.concurrency import MultiObject, MultiException, concurrent
from easypy.timing import repeat, timing
from easypy.bunch import Bunch
from easypy.units import Duration

from easypy.sync import iter_wait, wait, iter_wait_progress, Timer, TimeoutException, PredicateNotSatisfied
from easypy.sync import SynchronizationCoordinator, SYNC
from easypy.sync import shared_contextmanager
from easypy.sync import TagAlongThread
from easypy.sync import LoggedRLock, LockLeaseExpired
from easypy.sync import SynchronizedSingleton
from easypy.sync import LoggedCondition

from .test_logging import get_log  # noqa; pytest fixture


def test_shared_contextmanager():

    data = []

    @shared_contextmanager
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


def test_shared_contextmanager_method():

    class Foo(object):

        def __init__(self):
            self.data = []

        @shared_contextmanager
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


def test_shared_contextmanager_method_does_not_keep_object_alive_after_done():
    import weakref
    import gc

    class Foo:
        @shared_contextmanager
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


def verify_concurrent_order(executed, *expected):
    look_at_index = 0
    for expected_group in expected:
        executed_group = set(executed[look_at_index:look_at_index + len(expected_group)])
        assert executed_group == expected_group, 'wrong execution order'
        look_at_index += len(executed_group)
    assert look_at_index == len(executed), 'executed list is shorted than expected'


def test_synchronization_coordinator_wait_for_everyone():
    mo = MultiObject(range(3))

    sync = SynchronizationCoordinator(len(mo))
    executed = []

    def foo(i):
        def execute(caption):
            executed.append((i, caption))

        sleep(i / 10)
        execute('after sleep')
        sync.wait_for_everyone()
        execute('after wait')
        sync.wait_for_everyone()

        sleep(i / 10)
        execute('after sleep 2')
        sync.wait_for_everyone()
        execute('after wait 2')

    mo.call(foo)
    verify_concurrent_order(
        executed,
        {(i, 'after sleep') for i in range(3)},
        {(i, 'after wait') for i in range(3)},
        {(i, 'after sleep 2') for i in range(3)},
        {(i, 'after wait 2') for i in range(3)})


def test_synchronization_coordinator_collect_and_call_once():
    mo = MultiObject(range(3))

    sync = SynchronizationCoordinator(len(mo))
    executed = []

    def foo(i):
        def execute(caption):
            executed.append((i, caption))

        sleep(i / 10)

        def func_to_call_once(param):
            executed.append('params = %s' % sorted(param))
            return sum(param)
        result = sync.collect_and_call_once(i + 1, func_to_call_once)
        execute('result is %s' % result)

        assert sync.collect_and_call_once(i, len) == 3, 'parameters remain from previous call'

    mo.call(foo)
    verify_concurrent_order(
        executed,
        {'params = [1, 2, 3]'},
        {(i, 'result is 6') for i in range(3)})


def test_synchronization_coordinator_abandon():
    mo = MultiObject(range(3))

    sync = SynchronizationCoordinator(len(mo))
    executed = []

    def foo(i):
        def execute(caption):
            executed.append((i, caption))

        sync.wait_for_everyone()
        execute('after wait 1')

        if i == 2:
            sync.abandon()
            return
        # Only two waiters should reach here
        sync.wait_for_everyone()
        execute('after wait 2')

        # Even without explicit call to abandon, sync should only wait for two waiters
        sync.wait_for_everyone()
        execute('after wait 3')

    mo.call(foo)
    verify_concurrent_order(
        executed,
        {(i, 'after wait 1') for i in range(3)},
        {(i, 'after wait 2') for i in range(2)},
        {(i, 'after wait 3') for i in range(2)})


def test_synchronization_coordinator_exception_in_collect_and_call_once():
    mo = MultiObject(range(3))

    sync = SynchronizationCoordinator(len(mo))
    times_called = 0

    class MyException(Exception):
        pass

    def foo(i):
        def func_to_call_once(_):
            nonlocal times_called
            times_called += 1
            raise MyException

        with pytest.raises(MyException):
            sync.collect_and_call_once(i, func_to_call_once)

        assert sync.collect_and_call_once(i + 1, sum) == 6

    mo.call(foo)
    assert times_called == 1, 'collect_and_call_once with exception called the function more than once'


def test_synchronization_coordinator_with_multiobject():
    mo = MultiObject(range(3))

    executed = []

    def foo(i, _sync=SYNC):
        def execute(caption):
            executed.append((i, caption))

        sleep(i / 10)
        _sync.wait_for_everyone()
        execute('after wait')

        def func_to_call_once(param):
            executed.append('params = %s' % sorted(param))
            return sum(param)
        result = _sync.collect_and_call_once(i + 1, func_to_call_once)
        execute('result is %s' % result)

    foo(10)
    assert executed == [
        (10, 'after wait'),
        'params = [11]',
        (10, 'result is 11')]
    executed.clear()

    mo.call(foo)
    verify_concurrent_order(
        executed,
        {(i, 'after wait') for i in range(3)},
        {'params = [1, 2, 3]'},
        {(i, 'result is 6') for i in range(3)})


def test_synchronization_coordinator_with_multiobject_exception():
    mo = MultiObject(range(3))

    executed = []

    class MyException(Exception):
        pass

    def foo(i, _sync=SYNC):
        def execute(caption):
            executed.append((i, caption))

        _sync.wait_for_everyone()
        execute('after wait')

        if i == 2:
            raise MyException

        _sync.wait_for_everyone()
        execute('after wait/abandon')

    with pytest.raises(MultiException) as exc:
        mo.call(foo)
    assert exc.value.count == 1
    assert exc.value.common_type is MyException

    verify_concurrent_order(
        executed,
        {(i, 'after wait') for i in range(3)},
        {(i, 'after wait/abandon') for i in range(2)})


def test_synchronization_coordinator_with_multiobject_early_return():
    mo = MultiObject(range(3))

    executed = []

    def foo(i, _sync=SYNC):
        def execute(caption):
            executed.append((i, caption))

        _sync.wait_for_everyone()
        execute('after wait')

        if i == 2:
            return

        _sync.wait_for_everyone()
        execute('after wait/abandon')

    mo.call(foo)

    verify_concurrent_order(
        executed,
        {(i, 'after wait') for i in range(3)},
        {(i, 'after wait/abandon') for i in range(2)})


def test_synchronization_coordinator_with_multiobject_method():
    class Foo:
        def __init__(self, i):
            self.i = i

        def foo(self, _sync=SYNC):
            return (self.i, _sync.collect_and_call_once(self.i, lambda i_values: sorted(i_values)))

    mo = MultiObject(Foo(i) for i in range(3))

    assert mo.foo().T == (
        (0, [0, 1, 2]),
        (1, [0, 1, 2]),
        (2, [0, 1, 2]))


def test_synchronization_coordinator_failing_context_manager():
    class MyException(Exception):
        pass

    @contextmanager
    def foo(should_fail, _sync=SYNC):
        if should_fail:
            raise MyException()
        else:
            yield

    inside_executed = False
    with pytest.raises(MultiException[MyException]):
        with MultiObject([False, True]).call(foo):
            inside_executed = True

    assert not inside_executed, 'CM body executed even though __enter__ failed in one thread'


def test_synchronization_coordinator_timeout():
    mo = MultiObject(range(3))

    def foo(i, _sync=SYNC):
        sleep(i / 10)
        _sync.wait_for_everyone(timeout=0.1)

    with pytest.raises(MultiException) as exc:
        mo.call(foo)
    assert exc.value.count == len(mo)
    assert exc.value.common_type is threading.BrokenBarrierError


def test_synchronization_coordinator_with_context_manager():
    mo = MultiObject(range(3))

    executed = []

    @contextmanager
    def foo(i, _sync=SYNC):
        def execute(caption):
            executed.append((i, caption))

        sleep(i / 10)
        execute('after sleep')
        _sync.wait_for_everyone()
        execute('before yield')
        yield
        _sync.wait_for_everyone()
        execute('after yield')

    with mo.call(foo):
        executed.append('with body')

    verify_concurrent_order(
        executed,
        {(i, 'after sleep') for i in range(3)},
        {(i, 'before yield') for i in range(3)},
        {'with body'},
        {(i, 'after yield') for i in range(3)})


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
        with patch("easypy.sync._logger") as _logger:

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


def test_sync_singleton():

    class S(metaclass=SynchronizedSingleton):
        def __init__(self):
            sleep(1)

    a, b = MultiObject(range(2)).call(lambda _: S())
    assert a is b


def test_logged_condition():
    cond = LoggedCondition('test', log_interval=.1)

    progress = 0
    executed = []

    def wait_for_progress_to(progress_to):
        cond.wait_for(lambda: progress_to <= progress, 'progress to %s', progress_to)
        executed.append(progress_to)

    with concurrent(wait_for_progress_to, 10), concurrent(wait_for_progress_to, 20), concurrent(wait_for_progress_to, 30):
        with patch("easypy.sync._logger") as _logger:
            sleep(0.3)

        assert any(c == call("%s - waiting for progress to %s", cond, 10) for c in _logger.debug.call_args_list)
        assert any(c == call("%s - waiting for progress to %s", cond, 20) for c in _logger.debug.call_args_list)
        assert any(c == call("%s - waiting for progress to %s", cond, 30) for c in _logger.debug.call_args_list)
        assert executed == []

        with patch("easypy.sync._logger") as _logger:
            with cond.notifying_all('setting progress to 10'):
                progress = 10
        assert [c for c in _logger.debug.call_args_list if 'performed' in c[0][0]] == [
            call("%s - performed: setting progress to 10", cond)]

        with patch("easypy.sync._logger") as _logger:
            sleep(0.3)

        assert not any(c == call("%s - waiting for progress to %s", cond, 10) for c in _logger.debug.call_args_list)
        assert any(c == call("%s - waiting for progress to %s", cond, 20) for c in _logger.debug.call_args_list)
        assert any(c == call("%s - waiting for progress to %s", cond, 30) for c in _logger.debug.call_args_list)
        assert executed == [10]

        with patch("easypy.sync._logger") as _logger:
            with cond.notifying_all('setting progress to 30'):
                progress = 30
        assert [c for c in _logger.debug.call_args_list if 'performed' in c[0][0]] == [
            call("%s - performed: setting progress to 30", cond)]

        with patch("easypy.sync._logger") as _logger:
            sleep(0.3)

        assert not any(c == call("%s - waiting for progress to %s", cond, 10) for c in _logger.debug.call_args_list)
        assert not any(c == call("%s - waiting for progress to %s", cond, 20) for c in _logger.debug.call_args_list)
        assert not any(c == call("%s - waiting for progress to %s", cond, 30) for c in _logger.debug.call_args_list)
        assert executed == [10, 20, 30] or executed == [10, 30, 20]

        with patch("easypy.sync._logger") as _logger:
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


def test_wait_exception():
    with pytest.raises(Exception, match=".*`message` is required.*"):
        wait(0.1, pred=lambda: True)

    wait(0.1)
    wait(0.1, pred=lambda: True, message='message')
    wait(0.1, pred=lambda: True, message=False)
    repeat(0.1, callback=lambda: True)


def test_wait_better_exception():

    class TimedOut(PredicateNotSatisfied):
        pass

    i = 0

    def check():
        nonlocal i
        i += 1
        if i < 3:
            raise TimedOut(a=1, b=2)
        return True

    with pytest.raises(TimedOut):
        # due to the short timeout and long sleep, the pred would called exactly twice
        wait(.1, pred=check, sleep=1, message=False)

    assert i == 2
    wait(.1, pred=check, message=False)


def test_wait_better_exception_nested():

    class TimedOut(PredicateNotSatisfied):
        pass

    i = 0

    def check():
        nonlocal i
        i += 1
        if i < 3:
            raise TimedOut(a=1, b=2)
        return True

    with pytest.raises(TimedOut):
        # due to the short timeout and long sleep, the pred would called exactly twice
        # also, the external wait should call the inner one only once, due to the TimedOut exception,
        # which it knows not to swallow
        wait(5, lambda: wait(.1, pred=check, sleep=1, message=False), sleep=1, message=False)

    assert i == 2
    wait(.1, pred=check, message=False)


def test_iter_wait_warning():
    with pytest.raises(Exception, match=".*`message` is required.*"):
        for _ in iter_wait(0.1, pred=lambda: True):
            pass

    no_warn_iters = [
        iter_wait(0.1),
        iter_wait(0.1, pred=lambda: True, message='message'),
        iter_wait(0.1, pred=lambda: True, throw=False),
        iter_wait(0.1, pred=lambda: True, message=False)
    ]
    for i in no_warn_iters:
        for _ in i:
            pass


def test_iter_wait_progress_inbetween_sleep():
    data = Bunch(a=3)

    def get():
        data.a -= 1
        return data.a

    sleep = .1
    g = iter_wait_progress(get, advance_timeout=10, sleep=sleep)

    # first iteration should be immediate
    t = Timer()
    next(g)
    assert t.duration < sleep

    # subsequent iteration should be at least 'sleep' long
    next(g)
    assert t.duration >= sleep

    for state in g:
        pass
    assert state.finished is True


def test_iter_wait_progress_total_timeout():
    data = Bunch(a=1000)

    def get():
        data.a -= 1
        return data.a

    with pytest.raises(TimeoutException) as exc:
        for state in iter_wait_progress(get, advance_timeout=1, sleep=.05, total_timeout=.1):
            pass
    assert exc.value.message.startswith("advanced but failed to finish")


def test_wait_long_predicate():
    """
    After the actual check the predicate is held for .3 seconds. Make sure
    that we don't get a timeout after .2 seconds - because the actual
    condition should be met in .1 second!
    """

    t = Timer()

    def pred():
        try:
            return 0.1 < t.duration
        finally:
            wait(0.3)

    wait(0.2, pred, message=False)


def test_timeout_exception():
    exc = None

    with timing() as t:
        try:
            wait(0.5, lambda: False, message=False)
        except TimeoutException as e:
            exc = e

    assert exc.duration > 0.5
    assert exc.start_time >= t.start_time
    assert exc.start_time < t.stop_time
    assert t.duration > exc.duration


def test_wait_with_callable_message():
    val = ['FOO']

    with pytest.raises(TimeoutException) as e:
        def pred():
            val[0] = 'BAR'
            return False
        wait(pred=pred, timeout=.1, message=lambda: 'val is %s' % val[0])

    assert val[0] == 'BAR'
    assert e.value.message == 'val is BAR'


@pytest.mark.parametrize("multipred", [False, True])
def test_wait_do_something_on_final_attempt(multipred):
    data = []

    def pred(is_final_attempt):
        if is_final_attempt:
            data.append('final')
        data.append('regular')
        return False

    if multipred:
        pred = [pred]

    with pytest.raises(TimeoutException):
        wait(pred=pred, timeout=.5, sleep=.1, message=False)

    assert all(iteration == 'regular' for iteration in data[:-2])
    assert data[-2] == 'final'
    assert data[-1] == 'regular'


def test_wait_log_predicate(get_log):
    def pred():
        raise PredicateNotSatisfied('bad attempt')

    with pytest.raises(TimeoutException):
        wait(pred=pred, timeout=.5, sleep=.1, message=False, log_interval=0.2)
    durations = re.findall('Still waiting after (.*?): bad attempt', get_log())
    rounded_durations = [round(Duration(d), 2) for d in durations]
    assert rounded_durations == [0.2, 0.4], 'expected logs at 200ms and 400ms, got %s' % (durations,)
