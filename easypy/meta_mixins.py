from .meta import EasyMixin


class UninheritedDefaults(EasyMixin):
    """
    Fields declared in this class will be default in subclasses even if overwritten.


    >>> class Foo(UninheritedDefaults):
    >>>     a = 1
    >>> Foo.a
    1
    >>>
    >>> class Bar(Foo):
    >>>     a = 2
    >>> Bar.a
    2
    >>>
    >>> class Baz(Bar):
    >>>     pass
    >>> Baz.a  # gets the value from Foo, not from Bar
    1
    """

    def prepare(self):
        defaults = {k: v for k, v in self.orig_dct.items() if not k.startswith('__')}
        self.dct.update(defaults)

        @self.add_hook
        def before_subclass_init(name, bases, dct):
            for k, v in defaults.items():
                dct.setdefault(k, v)


