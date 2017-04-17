import pytest

from easypy.lockstep import lockstep, LockstepSyncMismatch


def test_lockstep_side_effects():
    calculation_result = 0

    @lockstep
    def simple_calculation(number):
        nonlocal calculation_result

        calculation_result = number
        yield 'SET_NUMBER'

        calculation_result *= 2
        yield 'MULTIPLY_IT_BY_TWO'

        calculation_result += 5
        yield 'ADD_FIVE'

    with simple_calculation(5) as calculation:
        calculation.step_next('SET_NUMBER')
        assert calculation_result == 5

        calculation.step_next('MULTIPLY_IT_BY_TWO')
        assert calculation_result == 10

        calculation.step_next('ADD_FIVE')
        assert calculation_result == 15


def test_lockstep_wrong_step_name():
    @lockstep
    def process():
        yield 'STEP_1'
        yield 'STEP_2'
        yield 'STEP_3'

    with pytest.raises(LockstepSyncMismatch) as excinfo:
        with process() as process:
            process.step_next('STEP_1')
            process.step_next('STEP_TWO')
            process.step_next('STEP_3')

    assert excinfo.value.expected_step == 'STEP_TWO'
    assert excinfo.value.actual_step == 'STEP_2'


def test_lockstep_not_exhausted():
    @lockstep
    def process():
        yield 'STEP_1'
        yield 'STEP_2'
        yield 'STEP_3'

    with pytest.raises(LockstepSyncMismatch) as excinfo:
        with process() as process:
            process.step_next('STEP_1')
            process.step_next('STEP_2')

    assert excinfo.value.expected_step == 'finished'
    assert excinfo.value.actual_step == 'STEP_3'


def test_lockstep_exhausted_prematurely():
    @lockstep
    def process():
        yield 'STEP_1'
        yield 'STEP_2'

    with pytest.raises(LockstepSyncMismatch) as excinfo:
        with process() as process:
            process.step_next('STEP_1')
            process.step_next('STEP_2')
            process.step_next('STEP_3')

    assert excinfo.value.expected_step == 'STEP_3'
    assert excinfo.value.actual_step == 'finished'


def test_lockstep_exhaust():
    finished = False

    @lockstep
    def process():
        nonlocal finished

        yield 'STEP_1'
        yield 'STEP_2'
        yield 'STEP_3'

        finished = True

    assert not finished
    with process() as process:
        assert not finished
        process.step_all()
        assert finished
    assert finished


def test_lockstep_yielded_values():
    @lockstep
    def process():
        yield 'STEP_1', 1
        yield 'STEP_2'
        yield 'STEP_3', 3

    with process() as process:
        assert process.step_next('STEP_1') == 1
        assert process.step_next('STEP_2') is None
        assert process.step_next('STEP_3') == 3


def test_lockstep_nested():
    @lockstep
    def internal_process():
        yield 'INTERNAL_1'
        yield 'INTERNAL_2'

    @lockstep
    def external_process():
        yield 'EXTERNAL_1'
        yield from internal_process()
        yield 'EXTERNAL_2'

    with external_process() as process:
        process.step_next('EXTERNAL_1')
        process.step_next('INTERNAL_1')
        process.step_next('INTERNAL_2')
        process.step_next('EXTERNAL_2')


def test_lockstep_step_util():
    @lockstep
    def process():
        yield 'STEP_1'
        yield 'STEP_2'
        yield 'STEP_3'

    with process() as process:
        process.step_until('STEP_3')


def test_lockstep_step_util_wrong_order():
    @lockstep
    def process():
        yield 'STEP_1'
        yield 'STEP_2'
        yield 'STEP_3'

    with pytest.raises(LockstepSyncMismatch) as excinfo:
        with process() as process:
            process.step_until('STEP_2')
            process.step_until('STEP_1')

    assert excinfo.value.expected_step == 'STEP_1'
    assert excinfo.value.actual_step == 'finished'
