import pytest

from contextlib import contextmanager

from easypy.signals import register_object, unregister_object, MissingIdentifier, PRIORITIES
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
