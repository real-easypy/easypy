import pytest

from easypy.meta import EasyMeta, EasyMixin


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


def test_easy_meta_before_cls_init():
    class Foo(metaclass=EasyMeta):
        @EasyMeta.Hook
        def before_subclass_init(name, bases, dct):
            try:
                base = dct.pop('CLASS')
            except KeyError:
                pass
            else:
                bases.insert(0, base)

    class Bar:
        pass

    class Baz(Foo):
        CLASS = Bar

        a = 1
        b = 2


    assert not hasattr(Baz, 'CLASS')
    assert issubclass(Baz, Bar)


def test_easy_mixin():
    class MyMixinCreator(EasyMixin):
        def prepare(self):
            verify = {k: v for k, v in self.orig_dct.items() if not k.startswith('__')}

            @self.add_hook
            def before_subclass_init(name, bases, dct):
                for k, v in verify.items():
                    assert v == dct[k], '%s is %s - needs to be %s' % (k, dct[k], v)

    class MyMixin(MyMixinCreator):
        a = 1
        b = 2

    class Foo(MyMixin):
        a = 1
        b = 2

    with pytest.raises(AssertionError) as exc:
        class Bar(MyMixin):
            a = 2
            b = 2

    assert 'a is 2 - needs to be 1' in (str(exc.value))


def test_uninherited_defaults():
    from easypy.meta_mixins import UninheritedDefaults

    class Defaults(UninheritedDefaults):
        a = 1
        b = 2

    class Foo(Defaults):
        a = 3
        c = 4

    class Bar(Foo):
        b = 5
        c = 6

    class Baz(Bar):
        pass


    assert Foo.a == 3
    assert Foo.b == 2
    assert Foo.c == 4

    assert Bar.a == 1, 'Bar.a should come from Defaults, not Foo'
    assert Bar.b == 5
    assert Bar.c == 6

    assert Baz.a == 1, 'Baz.a should come from Defaults, not Bar'
    assert Baz.b == 2, 'Baz.b should come from Defaults, not Bar'
    assert Baz.c == 6, 'Baz.c is not in Defaults so it should come from Bar'
