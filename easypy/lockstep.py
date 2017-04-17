from contextlib import contextmanager
from functools import wraps

from .exceptions import TException


class LockstepSyncMismatch(TException):
    template = "Expected {lockstep_name} to be {expected_step}, but it is {actual_step}"


class _LockstepInvocation(object):
    def __init__(self, name, generator):
        self._name = name
        self._generator = generator
        self._current_step = 'just-started'

    def step_next(self, step):
        """Progress to the next step, confirming it's the specified one"""

        try:
            self._current_step, value = self._next_step_and_value()
        except StopIteration:
            raise LockstepSyncMismatch(lockstep_name=self._name,
                                       expected_step=step,
                                       actual_step='finished')
        if self._current_step != step:
            raise LockstepSyncMismatch(lockstep_name=self._name,
                                       expected_step=step,
                                       actual_step=self._current_step)
        return value

    def step_until(self, step):
        """Progress until we get to the specified step, confirming that it exist"""

        while self._current_step != step:
            try:
                self._current_step, value = self._next_step_and_value()
            except StopIteration:
                raise LockstepSyncMismatch(lockstep_name=self._name,
                                           expected_step=step,
                                           actual_step='finished')
        return value

    def step_all(self):
        """Progress through all the remaining steps"""

        for self._current_step in self._generator:
            pass

    def _next_step_and_value(self):
        yield_result = next(self._generator)
        if isinstance(yield_result, tuple) and len(yield_result) == 2:
            return yield_result
        else:
            return yield_result, None

    def __str__(self):
        return '%s<%s>' % (self._name, self._current_step)


class _LockstepContextManagerWrapper(object):
    def __init__(self, cm):
        self._cm = cm

    def __enter__(self):
        return self._cm.__enter__()

    def __exit__(self, *args):
        return self._cm.__exit__(*args)

    def __iter__(self):
        with self._cm as invocation:
            yield from invocation._generator


def lockstep(generator_func):
    """
    Synchronize a coroutine that runs a process step-by-step.

    Decorate a generator that yields step names to create a context manager.
    The context object has a `.step_next`/`.step_until` methods that must be
    called, in order, with all expected step names, to make the generator
    progress to each step.
    """

    @contextmanager
    def cm(*args, **kwargs):
        invocation = _LockstepInvocation(generator_func.__name__, generator_func(*args, **kwargs))

        yield invocation

        try:
            step_not_taken, _ = invocation._next_step_and_value()
        except StopIteration:
            # all is well - all steps were exhausted
            invocation._current_step = 'finished'
        else:
            raise LockstepSyncMismatch(lockstep_name=invocation._name,
                                       expected_step='finished',
                                       actual_step=step_not_taken)

    @wraps(generator_func)
    def wrapper(*args, **kwargs):
        return _LockstepContextManagerWrapper(cm(*args, **kwargs))

    return wrapper
