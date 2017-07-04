import os
if os.getenv("GEVENT") == "true":
    from easypy.gevent import apply_patch
    apply_patch()


import logging
import logging.config
from easypy import logging as easypy_logging
logging.addLevelName(logging.WARN, "WARN")  # instead of "WARNING", so that it takes less space...
logging.addLevelName(logging.NOTSET, "NA")  # instead of "NOTSET", so that it takes less space...
if not issubclass(logging.Logger, easypy_logging.ContextLoggerMixin):
    logging.Logger.__bases__ = logging.Logger.__bases__ + (easypy_logging.ContextLoggerMixin,)


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s|%(process)2s:%(threadName)-25s|%(name)-40s|%(levelname)-5s|%(funcName)-30s |%(message)s')
