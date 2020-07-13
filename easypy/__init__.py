import os

# this is used internally, so we define it here, and copy it into .humanize
def yesno_to_bool(s):
    s = s.lower()
    if s not in ("yes", "no", "true", "false", "1", "0"):
        raise ValueError("Unrecognized boolean value: %r" % (s,))
    return s in ("yes", "true", "1")


gevent = os.getenv('EASYPY_AUTO_PATCH_GEVENT', '')
if gevent and yesno_to_bool(gevent):
    from easypy.gevent import apply_patch
    apply_patch()


framework = os.getenv('EASYPY_AUTO_PATCH_LOGGING', '')
if framework:
    from easypy.logging import initialize
    initialize(framework=framework, patch=True)
