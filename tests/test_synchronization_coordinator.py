import pytest

from time import sleep
from threading import BrokenBarrierError
from contextlib import contextmanager

from easypy.concurrency import MultiObject, MultiException
from easypy.concurrency import SynchronizationCoordinator, SYNC


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
        (2, [0, 1, 2]),
        )


def test_synchronization_coordinator_timeout():
    mo = MultiObject(range(3))

    def foo(i, _sync=SYNC):
        sleep(i / 10)
        _sync.wait_for_everyone(timeout=0.1)

    with pytest.raises(MultiException) as exc:
        mo.call(foo)
    assert exc.value.count == len(mo)
    assert exc.value.common_type is BrokenBarrierError


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
