import os
if os.getenv("GEVENT") == "true":
    from easypy.gevent import apply_patch
    apply_patch()

use_logbook = bool(os.getenv("LOGBOOK"))

if not use_logbook:
    import logging
    logging.addLevelName(logging.WARN, "WARN")  # instead of "WARNING", so that it takes less space...
    logging.addLevelName(logging.NOTSET, "NA")  # instead of "NOTSET", so that it takes less space...


from easypy.logging import initialize
initialize(patch=True, framework="logbook" if use_logbook else "logging", context=dict(domain='domain'))


if use_logbook:
    import logbook
    from logbook.handlers import StderrHandler
    from easypy.logging import ConsoleFormatter
    handler = StderrHandler(level=logbook.DEBUG)
    handler.formatter = ConsoleFormatter(
        "{record.extra[levelcolor]}<<"
        "{record.time:%Y-%m-%d %H:%M:%S}|"
        "{record.level_name:8}>>| "
        "{record.extra[domain]:15}| "
        "{record.extra[decoration]}{record.message}"
    )
    handler.push_application()
else:
    from logging import StreamHandler
    from easypy.logging import ConsoleFormatter
    formatter = ConsoleFormatter('%(levelcolor)s<<%(asctime)s|%(levelname)-5s|%(domain)-15s>>|%(decoration)s%(message)s', datefmt='%H:%M:%S')
    handler = StreamHandler()
    handler.setFormatter(formatter)
    logging.root.addHandler(handler)


import pytest


@pytest.fixture
def is_logbook():
    return use_logbook
