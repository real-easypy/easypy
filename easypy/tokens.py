# ===================================================================================================
# Module hack: ``from easypy.tokens import AUTO``
# ===================================================================================================
import sys
from types import ModuleType


class Token(str):

    _all = {}

    def __new__(cls, name):
        name = name.strip("<>")
        try:
            return cls._all[name]
        except KeyError:
            pass
        cls._all[name] = self = super().__new__(cls, "<%s>" % name)
        return self

    def __repr__(self):
        return self

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self is other
        elif isinstance(other, str):
            return self.strip("<>").lower() == other.strip("<>").lower()
        return False

    # we're already case insensitive when comparing
    def lower(self):
        return self

    def upper(self):
        return self

    def __hash__(self):
        return super().__hash__()


def if_auto(val, auto):
    """
    Convenience for `auto if val is AUTO else val`

    Example:

        config.foo_level = 100

        def foo(level=AUTO):
            level = if_auto(level, config.foo_level)
            return level

        assert foo() == 100
        assert foo(AUTO) == 100
        assert foo(1) == 1

    """
    AUTO = Token("AUTO")
    return auto if val is AUTO else val


class TokensModule(ModuleType):
    """The module-hack that allows us to use ``from easypy.tokens import AUTO``"""
    __all__ = ()  # to make help() happy
    __package__ = __name__
    _orig_module = sys.modules[__name__]

    def __getattr__(self, attr):
        try:
            return getattr(self._orig_module, attr)
        except AttributeError:
            pass

        if attr.startswith("_") or attr == 'trait_names':
            raise AttributeError(attr)

        token = Token("<%s>" % attr)
        setattr(self._orig_module, attr, token)
        return token

    def __dir__(self):
        return sorted(dir(self._orig_module) + list(Token._all))

    __path__ = []
    __file__ = __file__


mod = TokensModule(__name__, __doc__)
sys.modules[__name__] = mod

del ModuleType
del TokensModule
del mod, sys
