from easypy.meta import EasyMeta, GetAllSubclasses


def test_easy_meta_before_cls_init():

    class FooMaker(metaclass=EasyMeta):
        @EasyMeta.Hook
        def before_subclass_init(name, bases, dct):
            dct[name] = "foo"

    class BarMaker(metaclass=EasyMeta):
        @EasyMeta.Hook
        def before_subclass_init(name, bases, dct):
            dct[name] = "bar"

    class Foo(FooMaker):
        ...
    assert Foo.Foo == "foo"

    class Bar(BarMaker):
        ...
    assert Bar.Bar == "bar"

    class Baz1(FooMaker, BarMaker):
        ...
    assert Baz1.Baz1 == "bar"

    class Baz2(BarMaker, FooMaker):
        ...
    assert Baz2.Baz2 == "foo"


def test_easy_meta_after_cls_init():
    class Foo(metaclass=EasyMeta):
        @EasyMeta.Hook
        def after_subclass_init(cls):
            cls.foo_init = cls.__name__

    class Bar(metaclass=EasyMeta):
        @EasyMeta.Hook
        def after_subclass_init(cls):
            cls.bar_init = cls.__name__

    class Baz(Foo, Bar):
        @EasyMeta.Hook
        def after_subclass_init(cls):
            cls.baz_init = cls.__name__

    assert not hasattr(Foo, 'foo_init'), 'after_subclass_init declared in Foo invoked on Foo'
    assert not hasattr(Bar, 'bar_init'), 'after_subclass_init declared in Bar invoked on Bar'

    assert Baz.foo_init == 'Baz'
    assert Baz.bar_init == 'Baz'
    assert not hasattr(Baz, 'baz_init'), 'after_subclass_init declared in Baz invoked on Baz'


def test_easy_meta_get_all_subclasses():
    class Foo(GetAllSubclasses):
        pass

    class Bar(Foo):
        pass

    class Baz(Foo):
        pass

    class Qux(Bar):
        pass

    assert set(Foo.get_all_subclasses()) == {Bar, Baz, Qux}
    assert set(Bar.get_all_subclasses()) == {Qux}
    assert set(Baz.get_all_subclasses()) == set()
    assert set(Qux.get_all_subclasses()) == set()


def test_easy_meta_multi_inheritance():

    class A(metaclass=EasyMeta):
        @EasyMeta.Hook
        def before_subclass_init(name, bases, dct):
            dct.setdefault('name', []).append("A")

    class B(metaclass=EasyMeta):
        @EasyMeta.Hook
        def before_subclass_init(name, bases, dct):
            dct.setdefault('name', []).append("B")

    class AA(A):
        @EasyMeta.Hook
        def before_subclass_init(name, bases, dct):
            dct.setdefault('name', []).append("AA")

    class BB(B):
        @EasyMeta.Hook
        def before_subclass_init(name, bases, dct):
            dct.setdefault('name', []).append("BB")

    class A_B(A, B): ...
    class AA_BB(AA, BB): ...
    class A_BB(A, BB): ...
    class B_AA(B, AA): ...

    # reminder - hooks aren't invoked on the classes on which they're defined, only subclasses
    assert not hasattr(A, "name")
    assert not hasattr(B, "name")
    assert AA.name == ["A"]
    assert BB.name == ["B"]
    assert A_B.name == ["A", "B"]
    assert AA_BB.name == ["A", "AA", "B", "BB"]
    assert A_BB.name == ["A", "B", "BB"]
