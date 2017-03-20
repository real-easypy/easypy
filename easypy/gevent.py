try:
    from gevent.monkey import is_module_patched
except ImportError:
    def is_module_patched(*_, **__):
        return False


def patch_all():
    import gevent
    import threading
    gevent._orig_start_new_thread = threading._start_new_thread
    gevent.monkey.patch_all(Event=True, sys=True)
