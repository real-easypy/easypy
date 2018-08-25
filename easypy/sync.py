"""
This module is about synchronizing and coordinating between... things.
"""


from contextlib import contextmanager
import sys
from threading import RLock

from .decorations import wrapper_decorator
from .caching import locking_cache


@wrapper_decorator
def shared_contextmanager(func):

    @locking_cache
    def inner(*args, **kwargs):

        class CtxManager():
            def __init__(self):
                self.count = 0
                self.func_cm = contextmanager(func)
                self._lock = RLock()

            def __enter__(self):
                with self._lock:
                    if self.count == 0:
                        self.ctm = self.func_cm(*args, **kwargs)
                        self.obj = self.ctm.__enter__()
                    self.count += 1
                return self.obj

            def __exit__(self, *args):
                with self._lock:
                    self.count -= 1
                    if self.count > 0:
                        return
                    self.ctm.__exit__(*sys.exc_info())
                    del self.ctm
                    del self.obj

        return CtxManager()

    return inner
