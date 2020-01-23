import pytest

from contextlib import contextmanager

from easypy.signals import signal, register_object, unregister_object, MissingIdentifier, PRIORITIES
from easypy.signals import on_test, on_async_test
from easypy.signals import on_test_identifier
from easypy.signals import on_ctx_test
from easypy.signals import on_ctx_test_identifier

on_test_identifier.identifier = 'obj'
on_ctx_test_identifier.identifier = 'obj'
on_async_test.asynchronous = True


def test_simple_signal_registration():

    @on_test.register
    def foo(a, b):
        a / b

    with pytest.raises(ZeroDivisionError):
        on_test(a=5, b=0, c='c')

    on_test.unregister(foo)
    on_test(a=5, b=0, c='c')


def test_simple_signal_object_registration():

    class Foo():
        def on_test(self, a, b):
            a / b
        __init__ = register_object

    f = Foo()

    with pytest.raises(ZeroDivisionError):
        on_test(a=5, b=0, c='c')

    unregister_object(f)


def test_simple_signal_object_identifier():

    on_test_identifier(obj='xxx')

    class Foo():
        def on_test_identifier(self, a):
            a / self.b
        __init__ = register_object

    f1 = Foo()
    f2 = Foo()

    f1.b = 1
    f2.b = 0

    with pytest.raises(MissingIdentifier):
        on_test_identifier(a=5, b=0, c='c')

    on_test_identifier(a=5, b=0, obj=f1)

    with pytest.raises(ZeroDivisionError):
        on_test_identifier(a=5, b=0, obj=f2)

    unregister_object(f1)
    unregister_object(f2)

    with on_test_identifier.registered(f2.on_test_identifier):
        with pytest.raises(ZeroDivisionError):
            on_test_identifier(a=5, b=0, obj=f2)


def test_simple_signal_object_wo_identifier():

    class Foo():
        def on_test_identifier(self, obj):
            1 / 0
        __init__ = register_object

    f1 = Foo()

    with pytest.raises(ZeroDivisionError):
        on_test_identifier(a=5, b=0, obj='obj1')
        on_test_identifier(a=5, b=0, obj='obj2')

    unregister_object(f1)


def test_simple_signal_object_identifier_attribute():

    class Foo():
        def on_test_identifier(self):
            1 / 0
        __init__ = register_object

    f1 = Foo()
    f1.obj = 'obj1'

    on_test_identifier(obj='xxx')

    with pytest.raises(ZeroDivisionError):
        on_test_identifier(obj='obj1')

    unregister_object(f1)


def test_registration_context():

    def foo(a, b):
        a / b

    with on_test.registered(foo):
        with pytest.raises(ZeroDivisionError):
            on_test(a=5, b=0, c='c')

    on_test(a=5, b=0, c='c')


def test_priorities():

    l = []

    @on_test.register(priority=PRIORITIES.LAST)
    def last():
        l.append(3)

    @on_test.register(priority=PRIORITIES.FIRST)
    def first():
        l.append(1)

    @on_test.register()
    def mid():
        l.append(2)

    on_test()

    assert l == [1, 2, 3]


def test_priorities_async():

    l = []
    import time

    @on_test.register(asynchronous=True, priority=PRIORITIES.FIRST)
    def first():
        time.sleep(.05)
        l.append(1)

    @on_test.register(asynchronous=True, priority=PRIORITIES.LAST)
    def last():
        l.append(2)

    on_test()

    assert l == [1, 2]


def test_async():
    from threading import get_ident

    main = get_ident()

    @on_async_test.register(asynchronous=True)
    def a1():
        assert get_ident() != main

    @on_async_test.register()  # follows the current setting
    def a2():
        assert get_ident() != main

    with pytest.raises(AssertionError):
        @on_async_test.register(asynchronous=False)
        def xx():
            assert get_ident() == main

    on_async_test()


def test_ctx():
    result = []

    class Foo:
        @contextmanager
        def on_ctx_test(self, before, after):
            result.append(before)
            yield
            result.append(after)

    foo = Foo()

    register_object(foo)

    assert result == []
    with on_ctx_test(before=1, after=2):
        assert result == [1]
    assert result == [1, 2]

    unregister_object(foo)

    with on_ctx_test(before=3, after=4):
        assert result == [1, 2]
    assert result == [1, 2]


def test_ctx_with_identifier():
    class Foo():
        @contextmanager
        def on_ctx_test_identifier(self, before, after):
            self.result.append(before)
            yield
            self.result.append(after)

    f1 = Foo()
    f1.obj = 'obj1'
    f1.result = []
    f2 = Foo()
    f2.obj = 'obj2'
    f2.result = []

    register_object(f1)
    register_object(f2)

    with on_ctx_test_identifier(before=1, after=2, obj='obj1'):
        assert f1.result == [1]
        assert f2.result == []
    assert f1.result == [1, 2]
    assert f2.result == []

    with on_ctx_test_identifier(before=3, after=4, obj='obj2'):
        assert f1.result == [1, 2]
        assert f2.result == [3]
    assert f1.result == [1, 2]
    assert f2.result == [3, 4]


def test_signal_weakref():
    """
    Test that signals handlers of methods are deleted when their objects get collected
    """
    import gc

    class Foo:
        def on_test(self, a, b):
            a / b

    foo = Foo()
    register_object(foo)

    with pytest.raises(ZeroDivisionError):
        on_test(a=5, b=0, c='c')

    del foo
    gc.collect()

    on_test(a=5, b=0, c='c')


def test_signal_weakref_complex_descriptors():
    import gc
    from easypy.lockstep import lockstep

    class Foo:
        @lockstep
        def on_test(self, a, b):
            a / b

    foo = Foo()
    register_object(foo)

    with pytest.raises(ZeroDivisionError):
        on_test(a=5, b=0, c='c')

    del foo
    gc.collect()

    on_test(a=5, b=0, c='c')


def test_signal_weakref_context_manager_delete_after():
    import gc

    result = []

    class Foo:
        @contextmanager
        def on_ctx_test(self, before, after):
            result.append(before)
            yield
            result.append(after)

    foo = Foo()
    register_object(foo)

    with on_ctx_test(before=1, after=2):
        assert result == [1]
    assert result == [1, 2]

    del foo
    gc.collect()

    with on_ctx_test(before=3, after=4):
        assert result == [1, 2]
    assert result == [1, 2]


def test_signal_weakref_context_manager_delete_during():
    import gc

    result = []

    class Foo:
        @contextmanager
        def on_ctx_test(self, before, after):
            result.append(before)
            yield
            result.append(after)

    foo = Foo()
    register_object(foo)

    with on_ctx_test(before=1, after=2):
        assert result == [1]
    assert result == [1, 2]

    with on_ctx_test(before=3, after=4):
        assert result == [1, 2, 3]
        del foo
        gc.collect()
    # The context manager should keep the signal handler alive
    assert result == [1, 2, 3, 4]


def test_signal_weakref_with_identifier():
    import gc

    class Foo:
        def on_test_identifier(self, a, b):
            a / b

    foo = Foo()
    foo.obj = 'obj'
    register_object(foo)

    with pytest.raises(ZeroDivisionError):
        on_test_identifier(a=5, b=0, obj='obj')
    on_test_identifier(a=5, b=0, obj='noobj')

    del foo
    gc.collect()

    on_test_identifier(a=5, b=0, obj='obj')
    on_test_identifier(a=5, b=0, obj='noobj')


def test_signal_weakref_context_manager_delete_after_with_identifier():
    import gc

    result = []

    class Foo:
        @contextmanager
        def on_ctx_test_identifier(self, before, after):
            result.append(before)
            yield
            result.append(after)

    foo = Foo()
    foo.obj = 'obj'
    register_object(foo)

    foo2 = Foo()
    foo2.obj = 'otherobj'
    register_object(foo2)

    with on_ctx_test_identifier(before=1, after=2, obj='obj'):
        assert result == [1]
    assert result == [1, 2]

    del foo
    del foo2
    gc.collect()

    with on_ctx_test_identifier(before=3, after=4, obj='obj'):
        assert result == [1, 2]
    assert result == [1, 2]


def test_signal_weakref_context_manager_delete_during_with_identifier():
    import gc

    result = []

    class Foo:
        @contextmanager
        def on_ctx_test_identifier(self, before, after):
            result.append(before)
            yield
            result.append(after)

    foo = Foo()
    foo.obj = 'obj'
    register_object(foo)

    foo2 = Foo()
    foo2.obj = 'otherobj'
    register_object(foo2)

    with on_ctx_test_identifier(before=1, after=2, obj='obj'):
        assert result == [1]
    assert result == [1, 2]

    with on_ctx_test_identifier(before=3, after=4, obj='obj'):
        assert result == [1, 2, 3]
        del foo
        del foo2
        gc.collect()
    # The context manager should keep the signal handler alive
    assert result == [1, 2, 3, 4]


def test_signal_decorator():
    @signal
    def on_test_signal_decorator(value): ...

    result = {}

    on_test_signal_decorator.register(lambda value: result.update(a=value))
    on_test_signal_decorator.register(lambda value: result.update(b=value))

    on_test_signal_decorator(value=1)
    assert result == {'a': 1, 'b': 1}

    on_test_signal_decorator(value=2)
    assert result == {'a': 2, 'b': 2}


def test_signal_decorator_contextmanager():
    result = []

    @signal
    @contextmanager
    def on_test_signal_decorator_contextmanager(before, after):
        yield

    @on_test_signal_decorator_contextmanager.register
    @contextmanager
    def handler1(before, after):
        result.append(before)
        yield
        result.append(after)

    assert result == []
    with on_test_signal_decorator_contextmanager(before=1, after=2):
        assert result == [1]
    assert result == [1, 2]


def test_signal_decorator_type_verification():
    @signal
    def on_test_signal_decorator_type_verification(a, b: int, c): ...

    on_test_signal_decorator_type_verification(a=1, b=2, c=3)
    on_test_signal_decorator_type_verification(a='hi', b=2, c='bye')

    with pytest.raises(TypeError) as err:
        on_test_signal_decorator_type_verification()
    assert 'missing' in str(err.value)

    with pytest.raises(TypeError) as err:
        on_test_signal_decorator_type_verification(a=1, b='two', c=3)
    assert 'Expected b:' in str(err.value)

    with pytest.raises(TypeError) as err:
        on_test_signal_decorator_type_verification(a=1, b=2, c=3, d=4)
    assert 'unexpected keyword argument' in str(err.value)

    @on_test_signal_decorator_type_verification.register
    def all_arguments(a, b, c):
        pass

    @on_test_signal_decorator_type_verification.register
    def less_arguments_in_different_order(c, a):
        pass

    with pytest.raises(TypeError) as err:
        @on_test_signal_decorator_type_verification.register
        def var_args(*args):
            pass
    assert 'illegal parameters' in str(err.value)

    with pytest.raises(TypeError) as err:
        @on_test_signal_decorator_type_verification.register
        def extra_parameters(a, b, c, d):
            pass
    assert 'parameters not in signal' in str(err.value)

    @on_test_signal_decorator_type_verification.register
    def extra_parameters_with_default(a, b, c, d=1):
        pass

    @on_test_signal_decorator_type_verification.register
    def extra_parameters_kwargs(a, b, c, **kwargs):
        pass

    on_test_signal_decorator_type_verification(a=1, b=2, c=3)


def test_signal_handler_registration_from_object_for_decorated_signals():
    from easypy.signals import on_test as on_test_by_import

    @signal
    def on_test(): ...

    assert on_test is not on_test_by_import

    class Foo:
        def __init__(self):
            self.result = []

        def on_test__by_import(self):
            self.result.append('i')

        @on_test.handler
        def on_test__by_decoration(self):
            self.result.append('d')


    foo = Foo()
    register_object(foo)

    assert foo.result == []

    on_test_by_import()
    assert foo.result == ['i']

    on_test()
    assert foo.result == ['i', 'd']

    unregister_object(foo)

    on_test_by_import()
    assert foo.result == ['i', 'd']

    on_test()
    assert foo.result == ['i', 'd']


def test_signal_handler_marking_methods_with_wrong_signature():
    @signal
    def my_signal(a, b: int): ...

    with pytest.raises(TypeError) as err:
        @my_signal.handler
        def no_args_handler():
            pass
    assert 'has no arguments' in str(err.value)

    with pytest.raises(TypeError) as err:
        @my_signal.handler
        def no_args_handler(a, b):
            pass
    assert 'not the `self` argument' in str(err.value)

    with pytest.raises(TypeError) as err:
        class Foo:
            @my_signal.handler
            def my_handler(self, d):
                pass
    assert 'parameters not in signal' in str(err.value)
