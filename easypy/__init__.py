# ===================================================================================================
# Module hack: ``from easypy.tokens import AUTO``
# ===================================================================================================
import sys
from types import ModuleType
from uuid import uuid4, UUID
from weakref import WeakKeyDictionary


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


MAIN_UUID = UUID(int=0)
UUIDS_TREE = WeakKeyDictionary()
UUID_TO_IDENT = WeakKeyDictionary()
IDENT_TO_UUID = {}


def _set_thread_uuid(ident, parent_uuid=MAIN_UUID):
    uuid = uuid4()
    IDENT_TO_UUID[ident] = uuid
    UUIDS_TREE[uuid] = parent_uuid


def _set_main_uuid():
    import threading
    IDENT_TO_UUID[threading.main_thread().ident] = MAIN_UUID
    UUID_TO_IDENT[MAIN_UUID] = threading.main_thread().ident


_set_main_uuid()
