from contextlib import contextmanager
from functools import wraps, update_wrapper

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

    def __iter__(self):
        yield from self._generator
        self._current_step = 'finished'

    def _next_step_and_value(self):
        yield_result = next(self._generator)
        if isinstance(yield_result, tuple) and len(yield_result) == 2:
            return yield_result
        else:
            return yield_result, None

    def __str__(self):
        return '%s<%s>' % (self._name, self._current_step)


class lockstep(object):
    """
    Synchronize a coroutine that runs a process step-by-step.

    Decorate a generator that yields step names to create a context manager.

    * Use like a regular method(that returns `None`).
    * Use `.lockstep(...)` on the method to get a context manager. The context
      object has a `.step_next`/`.step_until` methods that must be called, in
      order, with all expected step names, to make the generator progress to
      each step.
    * Yield from the context object to embed the lockstep inside a bigger
      lockstep function.

    Example:

        @lockstep
        def my_process():
            # things before step A
            yield 'A'
            # things between step A and step B
            yield 'B'
            # things between step B and step C
            yield 'C'
            # things between step C and step D
            yield 'D'
            # things after step D

        my_process()  # just run it like a normal function, ignoring the lockstep

        with my_process.lockstep() as process:
            process.step_next('A')  # go to next step - A
            process.step_until('C')  # go through steps until you reach C
            process.step_all()  # go through all remaining steps until the end

        @lockstep
        def bigger_process():
            yield 'X'

            # Embed `my_process`'s steps inside `bigger_process`:
            with my_process.lockstep() as process:
                yield from process

            yield 'Y'
    """

    def __init__(self, generator_func, _object=None):
        self.generator_func = generator_func
        self._object = _object
        update_wrapper(self, generator_func)

    @contextmanager
    def lockstep(self, *args, **kwargs):
        if self._object is None:
            generator = self.generator_func(*args, **kwargs)
        else:
            generator = self.generator_func(self._object, *args, **kwargs)
        invocation = _LockstepInvocation(self.generator_func.__name__, generator)

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

    def __call__(self, *args, **kwargs):
        with self.lockstep(*args, **kwargs) as process:
            process.step_all()

    def __get__(self, obj, _=None):
        if obj is None:
            return self

        return lockstep(self.generator_func, _object=obj)
