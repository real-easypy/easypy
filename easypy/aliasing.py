from itertools import chain


def super_dir(obj):
    """
    A python2/3 compatible way of getting the default ('super') behavior of __dir__
    """
    return sorted(set(chain(dir(type(obj)), obj.__dict__)))


class AliasingMixin():
    @property
    def _aliased(self):
        try:
            return getattr(self, self._ALIAS)
        except AttributeError:
            raise RuntimeError("object %r does no contain aliased object %r" % (self, self._ALIAS))

    def __dir__(self):
        members = set(super_dir(self))
        members.update(n for n in dir(self._aliased) if not n.startswith("_"))
        return sorted(members)

    def __getattr__(self, attr):
        if attr.startswith("_"):
            raise AttributeError(attr)
        return getattr(self._aliased, attr)


def aliases(name, static=True):
    """
    A class decorator that makes objects of a class delegate to an object they contain.
    Inspired by D's "alias this".

    Example:

        class B():
            def foo(self):
                print('foo')

        @aliases('b')
        class A():
            b = B()

        a = A()
        a.foo()  # delegated to b.foo()


        @aliases('b', static=False)
        class A():
            def __init__(self):
                b = B()

        a = A()
        a.foo()  # delegated to b.foo()

    """
    def deco(cls):
        assert not static or hasattr(cls, name)
        cls._ALIAS = name
        cls.__bases__ = cls.__bases__ + (AliasingMixin, )
        return cls
    return deco
