import os
if os.getenv("GEVENT") == "true":
    from easypy.gevent import patch_all
    patch_all()
