"""
Common initialization that needs to be called before any easypy module that deals with multithreading
"""

from uuid import uuid4, UUID
from weakref import WeakKeyDictionary


MAIN_UUID = UUID(int=0)
UUIDS_TREE = WeakKeyDictionary()
UUID_TO_IDENT = WeakKeyDictionary()
IDENT_TO_UUID = {}
_BOOTSTRAPPERS = set()


def _set_thread_uuid(ident, parent_uuid=MAIN_UUID):
    uuid = uuid4()
    IDENT_TO_UUID[ident] = uuid
    UUIDS_TREE[uuid] = parent_uuid


def _set_main_uuid():
    import threading
    IDENT_TO_UUID[threading.main_thread().ident] = MAIN_UUID
    UUID_TO_IDENT[MAIN_UUID] = threading.main_thread().ident


def get_thread_uuid(thread=None):
    """
    Assigns and returns a UUID to our thread, since thread.ident can be recycled.
    The UUID is used in mapping child threads to their parent threads.
    """
    import threading
    if not thread:
        thread = threading.current_thread()

    ident = thread.ident
    try:
        uuid = IDENT_TO_UUID[ident]
    except KeyError:
        uuid = IDENT_TO_UUID.setdefault(ident, uuid4())
        UUID_TO_IDENT[uuid] = ident
    return uuid


_set_main_uuid()
