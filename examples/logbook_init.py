import os
import logbook
from easypy.logging import initialize


LOG_PATH = "."
PRIMARY_LOG_FILE = os.path.join(LOG_PATH, 'easypy.log')
LOG_LEVEL = "INFO"

DETAILED = (
    '{record.asctime}|{process:2}:{thread_name:25}|{name:40}|{levelname:5}|'
    '{funcName:30} |{host:35}|{message}'
)

CONSOLE = (
    "{record.time:%Y-%m-%d %H:%M:%S}|{record.channel:8}|"
    "{record.level_name}<<{record.level_name:8}>>|"
    "{record.extra[domain]:10}| "
    "{record.extra[decoration]}{record.message}"
)


_action = 0


def handle_usr_signal(sig, frame):
    extra = dict(host="---")

    def dump_stacks():
        from easypy.threadtree import get_thread_stacks
        stacks = get_thread_stacks()
        logbook.info("\n%s", stacks, extra=extra)

    actions = {0: dump_stacks, 1: set_verbose, 2: set_info}

    global _action
    func = actions[_action % len(actions)]
    _action += 1
    logbook.info("YELLOW<<signal caught (%s)>> -> CYAN<<%s>>", _action, func, extra=extra)
    try:
        func()
    except:
        pass


def configure(filename=None, no_heartbeats=False):
    import socket
    global get_console_handler

    HOSTNAME = socket.gethostname()
    initialize(context=dict(domain=HOSTNAME), framework="logbook", patch=True)
    from easypy.logging import get_console_handler, ConsoleFormatter
    from easypy.colors import register_colorizers
    register_colorizers(
        CRITICAL='red',
        ERROR='red',
        WARNING='yellow',
        NOTICE='white',
        TRACE='dark_gray'
    )
    from logbook.handlers import StderrHandler
    handler = StderrHandler(level=LOG_LEVEL)
    handler.formatter = ConsoleFormatter(CONSOLE)
    handler.push_application()

    import signal
    signal.signal(signal.SIGUSR2, handle_usr_signal)


def set_level(level):
    get_console_handler().level = logbook.lookup_level(level)


def get_level():
    return logbook.get_level_name(get_console_handler().level)


def set_verbose():
    set_level("DEBUG")


def set_info():
    set_level("INFO")


if __name__ == "__main__":
    configure()

    from easypy.logging import EasypyLogger
    from easypy.concurrency import MultiObject
    from time import sleep
    import random

    def log(_, depth=0):
        logger = EasypyLogger("easypy.test.%s" % depth)
        for level in range(logbook.TRACE, logbook.CRITICAL + 1):
            with logger.indented("Level is: %s", level):
                logger.info("hey")
                logger.info("GREEN<<check>>")
                logger.warning("YELLOW<<warning>>")
                logger.error("error!")
                sleep(random.random() * .2)
                logger.trace("ping")
                if random.random() > 0.97:
                    log(depth + 1)

            set_level(level)

    MultiObject("abcde", log_ctx=lambda x: dict(domain=x)).call(log)
