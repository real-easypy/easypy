from easypy.meta import EasyMeta


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
