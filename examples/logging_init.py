import os
import logging
import logging.config
import logging.handlers

from easypy.logging import initialize
from easypy.logging import get_console_handler

logging.addLevelName(logging.WARN, "WARN")  # instead of "WARNING", so that it takes less space...
logging.addLevelName(logging.NOTSET, "NA")  # instead of "NOTSET", so that it takes less space...


LOG_PATH = "."
PRIMARY_LOG_FILE = os.path.join(LOG_PATH, 'easypy.log')
LOG_LEVEL = "INFO"

if os.getenv('TERM_LOG_STDOUT'):
    console_stream = 'stdout'
else:
    console_stream = 'stderr'


CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'detailed': {
            'format': '%(asctime)s|%(process)2s:%(threadName)-25s|%(name)-40s|%(levelname)-5s|'
                      '%(funcName)-30s |%(host)-35s|%(message)s',
        },
        'console': {
            '()': 'easypy.logging.ConsoleFormatter',
            'fmt': '%(levelcolor)s<<%(asctime)s|%(levelname)-5s|%(host)-40s>>|%(decoration)s%(message)s',
            'datefmt': '%H:%M:%S'
        },
        'yaml': {
            '()': 'easypy.logging.YAMLFormatter',
            'allow_unicode': True,
            'explicit_start': True,
            'explicit_end': True,
            'encoding': 'utf-8',
        },
    },
    'filters': {
        'thread_control': {
            '()': 'easypy.logging.ThreadControl'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': "console",  # if sys.stdout.isatty() else "detailed",
            'filters': ['thread_control'],
            'level': LOG_LEVEL,
            'stream': 'ext://sys.%s' % console_stream
        },
        'main_file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': PRIMARY_LOG_FILE,
            'mode': 'w',
            'formatter': 'detailed',
            'level': 'DEBUG',
            'maxBytes': 2**20 * 10,
            'backupCount': 5,
            'delay': True,
            'encoding': 'utf-8',
        },
        'aux': {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': os.path.join(LOG_PATH, 'aux.log'),
            'mode': 'w',
            'formatter': 'detailed',
            'level': 'DEBUG',
            'maxBytes': 2**20 * 10,
            'backupCount': 5,
            'delay': True,
            'encoding': 'utf-8',
        },
        'boto': {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': os.path.join(LOG_PATH, 'boto.log'),
            'mode': 'w',
            'formatter': 'detailed',
            'level': 'DEBUG',
            'maxBytes': 2**20 * 10,
            'backupCount': 5,
            'delay': True,
            'encoding': 'utf-8',
        },
        'threads': {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': os.path.join(LOG_PATH, 'threads.yaml.log'),
            'mode': 'w',
            'formatter': 'yaml',
            'level': 'DEBUG',
            'maxBytes': 2**20 * 100,
            'backupCount': 1,
            'delay': True,
            'encoding': 'utf-8',
        },
        'gevent': {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': os.path.join(LOG_PATH, 'gevent.log'),
            'mode': 'w',
            'formatter': 'detailed',
            'level': 'DEBUG',
            'maxBytes': 2**20 * 1,
            'backupCount': 1,
            'delay': True,
            'encoding': 'utf-8',
        },
        'progress': {
            'class': 'easypy.logging.ProgressHandler',
        },
    },
    'root': {
        'level': 'NOTSET',
        'handlers': ['console', 'main_file', 'progress']
    },
    'loggers': {
        'threads': {
            'propagate': False,
            'handlers': ['threads']
        },
        'gevent': {
            'propagate': False,
            'handlers': ['gevent']
        },
    }
}


REDIRECTIONS = {
    'aux': [
        'paramiko', 'paramiko.transport', 'plumbum.shell', 'urllib3.connectionpool', 'urllib3.util.retry',
        'aux', 'requests.packages.urllib3.connectionpool', 'googleapiclient.discovery',
        'sentry.errors', 'wepy.devops.talker.verbose', 'easypy.threadtree', 'concurrent.futures',
        'easypy.concurrency.locks',
    ],
    'boto': ['boto', 'boto3', 'botocore']
}


for target, loggers in REDIRECTIONS.items():
    for name in loggers:
        CONFIG['loggers'][name] = dict(propagate=False, handlers=[target, 'progress'])


_action = 0


def handle_usr_signal(sig, frame):
    extra = dict(host="---")

    def dump_stacks():
        from easypy.threadtree import get_thread_stacks
        stacks = get_thread_stacks()
        logging.info("\n%s", stacks, extra=extra)

    actions = {0: dump_stacks, 1: set_verbose, 2: set_info}

    global _action
    func = actions[_action % len(actions)]
    _action += 1
    logging.info("YELLOW<<signal caught (%s)>> -> CYAN<<%s>>", _action, func, extra=extra)
    try:
        func()
    except:
        pass


def configure(filename=None, no_heartbeats=False):
    import socket

    HOSTNAME = socket.gethostname()
    initialize(context=dict(host=HOSTNAME))

    import signal
    signal.signal(signal.SIGUSR2, handle_usr_signal)

    logging.config.dictConfig(CONFIG)


def set_level(level):
    get_console_handler().setLevel(getattr(logging, level))


def get_level():
    return logging.getLevelName(get_console_handler().level)


def set_verbose():
    set_level("DEBUG")


def set_info():
    set_level("INFO")
