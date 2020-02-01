from collections import defaultdict
from functools import wraps
from inspect import isgeneratorfunction, signature, Parameter
from contextlib import contextmanager as orig_contextmanager
from contextlib import ExitStack
from contextlib import _GeneratorContextManager


class KeyedStack(ExitStack):
    def __init__(self, context_factory):
        self.context_factory = context_factory
        self.contexts_dict = defaultdict(list)
        super().__init__()

    def enter_context(self, *key):
        cm = self.context_factory(*key)
        self.contexts_dict[key].append(cm)
        super().enter_context(cm)

    def exit_context(self, *key):
        self.contexts_dict[key].pop(-1).__exit__(None, None, None)


class _BetterGeneratorContextManager(_GeneratorContextManager):
    "This helper can handle generators and other context-managers"

    def __call__(self, func):
        if isgeneratorfunction(func):
            def inner(*args, **kwds):
                with self._recreate_cm():
                    yield from func(*args, **kwds)
        elif is_contextmanager(func):
            @contextmanager
            def inner(*args, **kwds):
                with self._recreate_cm():
                    with func(*args, **kwds) as ret:
                        yield ret
        else:
            def inner(*args, **kwds):
                with self._recreate_cm():
                    return func(*args, **kwds)
        return wraps(func)(inner)


# Some python version have a different signature for '_GeneratorContextManager.__init__', so we must adapt:
if signature(_GeneratorContextManager).parameters['args'].kind is Parameter.VAR_POSITIONAL:
    def contextmanager(func):
        @wraps(func)
        def helper(*args, **kwds):
            return _BetterGeneratorContextManager(func, *args, **kwds)
        return helper
else:
    def contextmanager(func):
        @wraps(func)
        def helper(*args, **kwds):
            return _BetterGeneratorContextManager(func, args, kwds)
        return helper

contextmanager.__doc__ = """@contextmanager decorator.

    Typical usage::

        @contextmanager
        def ctx(<arguments>):
            <setup>
            try:
                yield <value>
            finally:
                <cleanup>

    In a ``with`` statement::

        with ctx(<arguments>) as <variable>:
            <body>


    As a decorator for a function/method::

        @ctx(<arguments>)
        def simple_function():
            <do-something>

    As a decorator for a generator::

        @ctx(<arguments>)
        def generator():
            yield <something>

    As a decorator for a context manager (only those created using the @contextmanager decorator)::

        @ctx(<arguments>)
        @contextmanager
        def some_other_context_manager():
            <setup>
            try:
                yield <value>
            finally:
                <cleanup>
"""


# we use these to identify functions decorated by 'contextmanager'
_ctxm_code_samples = {
    f(None).__code__ for f in
    [contextmanager, orig_contextmanager]}


def is_contextmanager(func):
    return getattr(func, "__code__", None) in _ctxm_code_samples


@contextmanager
def breakable_section():
    """
    Useful for getting out of some deep nesting, as an alternative to a closure:

        item = None
        with breakable_section() as Break:
            if alpha:
                item = alpha.value
                raise Break

            if beta:
                for opt in beta.items:
                    if opt.is_the_one:
                        item = opt.value
                        raise Break

    Note that each 'Break' class this context-manager yield is unique,
    i.e it will only be caught by the context-manager that created it:

        with breakable_section() as Break1:

            with breakable_section() as Break2:
                raise Break1

            assert False  # will not reach here
    """
    Break = type("Break", (Exception,), {})
    try:
        yield Break
    except Break:
        pass
