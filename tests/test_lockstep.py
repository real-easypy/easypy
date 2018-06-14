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

    with simple_calculation.lockstep(5) as calculation:
        calculation.step_next('SET_NUMBER')
        assert calculation_result == 5

        calculation.step_next('MULTIPLY_IT_BY_TWO')
        assert calculation_result == 10

        calculation.step_next('ADD_FIVE')
        assert calculation_result == 15


def test_lockstep_run_as_function():
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

    simple_calculation(10)
    assert calculation_result == 25


def test_lockstep_class_method():
    class SimpleCalculation():
        def __init__(self, number):
            self.calculation_result = number

        @lockstep
        def calculation(self):
            self.calculation_result *= 2
            yield 'MULTIPLY_IT_BY_TWO'

            self.calculation_result += 5
            yield 'ADD_FIVE'

    simple_calculation = SimpleCalculation(5)
    with simple_calculation.calculation.lockstep() as calculation:
        assert simple_calculation.calculation_result == 5

        calculation.step_next('MULTIPLY_IT_BY_TWO')
        assert simple_calculation.calculation_result == 10

        calculation.step_next('ADD_FIVE')
        assert simple_calculation.calculation_result == 15
    assert simple_calculation.calculation_result == 15

    # run as function
    simple_calculation2 = SimpleCalculation(10)
    assert simple_calculation2.calculation_result == 10
    simple_calculation2.calculation()
    assert simple_calculation2.calculation_result == 25


def test_lockstep_wrong_step_name():
    @lockstep
    def process():
        yield 'STEP_1'
        yield 'STEP_2'
        yield 'STEP_3'

    with pytest.raises(LockstepSyncMismatch) as excinfo:
        with process.lockstep() as process:
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
        with process.lockstep() as process:
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
        with process.lockstep() as process:
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
    with process.lockstep() as process:
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

    with process.lockstep() as process:
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
        with internal_process.lockstep() as process:
            yield from process
        yield 'EXTERNAL_2'

    with external_process.lockstep() as process:
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

    with process.lockstep() as process:
        process.step_until('STEP_3')


def test_lockstep_step_util_wrong_order():
    @lockstep
    def process():
        yield 'STEP_1'
        yield 'STEP_2'
        yield 'STEP_3'

    with pytest.raises(LockstepSyncMismatch) as excinfo:
        with process.lockstep() as process:
            process.step_until('STEP_2')
            process.step_until('STEP_1')

    assert excinfo.value.expected_step == 'STEP_1'
    assert excinfo.value.actual_step == 'finished'


def test_lockstep_as_static_and_class_methods():
    class Foo:
        @lockstep
        def process1(self, out):
            out.append(1)
            yield 'STEP'
            out.append(2)

        @lockstep
        @classmethod
        def process2(cls, out):
            out.append(1)
            yield 'STEP'
            out.append(2)

        @lockstep
        @staticmethod
        def process3(out):
            out.append(1)
            yield 'STEP'
            out.append(2)

    print()

    def check_method_call(method):
        out = []
        method(out)
        assert out == [1, 2]

    check_method_call(Foo().process1)
    check_method_call(Foo().process2)
    check_method_call(Foo().process3)

    def check_method_lockstep(method):
        method.lockstep
        out = []
        with method.lockstep(out) as process:
            assert out == []
            process.step_until('STEP')
            assert out == [1]
        assert out == [1, 2]

    check_method_lockstep(Foo().process1)
    check_method_lockstep(Foo.process2)
    check_method_lockstep(Foo.process3)
