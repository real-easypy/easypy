import os
import logging
import logging.config
import os
from easypy import logging as easypy_logging

if os.getenv("GEVENT") == "true":
    from easypy.gevent import patch_all
    patch_all()


logging.addLevelName(logging.WARN, "WARN")  # instead of "WARNING", so that it takes less space...
logging.addLevelName(logging.NOTSET, "NA")  # instead of "NOTSET", so that it takes less space...
if not issubclass(logging.Logger, easypy_logging.ContextLoggerMixin):
    logging.Logger.__bases__ = logging.Logger.__bases__ + (easypy_logging.ContextLoggerMixin,)


# logging.basicConfig(level=logging.DEBUG)
