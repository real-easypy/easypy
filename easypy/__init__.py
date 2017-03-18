# ===================================================================================================
# Module hack: ``from easypy.tokens import AUTO``
# ===================================================================================================
import sys
from types import ModuleType


class TokensModule(ModuleType):
    """The module-hack that allows us to use ``from easypy.tokens import AUTO``"""
    __all__ = ()  # to make help() happy
    __package__ = __name__

    def __getattr__(self, attr):
        if attr.startswith("_") or attr == 'trait_names':
            raise AttributeError(attr)
        from .misc import Token
        token = Token("<%s>" % attr)
        setattr(self, attr, token)
        return token

    __path__ = []
    __file__ = __file__

mod = TokensModule(__name__ + ".tokens", TokensModule.__doc__)
sys.modules[mod.__name__] = mod

del ModuleType
del TokensModule
del mod, sys
