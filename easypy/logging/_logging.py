# encoding: utf-8
from __future__ import absolute_import

import logging
import os
import threading
from itertools import cycle, chain, repeat, count
from collections import OrderedDict

from easypy.colors import colorize, uncolored
from easypy.threadtree import ThreadContexts
from easypy.contexts import contextmanager

from easypy.logging import G, get_level_color, THREAD_LOGGING_CONTEXT


CLEAR_EOL = '\x1b[0K'


LEVEL_COLORS = {
    logging.DEBUG: "DARK_GRAY",
    logging.INFO: "GRAY",
    logging.WARNING: "YELLOW",
    logging.ERROR: "RED",
    logging.INFO + 1: "WHITE",
}


class LogLevelClamp(logging.Filterer):
    """
    Log-records with a log-level that is too high are clamped down.
    Used internally by the ``ProgressBar``
    """

    def __init__(self, level=logging.DEBUG):
        self.level = level
        self.name = logging.getLevelName(level)

    def filter(self, record):
        if record.levelno > self.level:
            record.levelname, record.levelno = self.name, self.level
        return True

    def __enter__(self):
        self.addFilter(self)

    def __exit__(self, *args):
        self.removeFilter(self)


def get_console_handler():
    try:
        return logging._handlers['console']
    except KeyError:
        for handler in logging.root.handlers:
            if not isinstance(handler, logging.StreamHandler):
                continue
            return handler


class ThreadControl(logging.Filter):
    """
    Used by ContextLoggerMixin .solo and .suppressed methods to control logging to console
    To use, add it to the logging configuration as a filter in the console handler

        ...
        'filters': {
            'thread_control': {
                '()': 'easypy.logging.ThreadControl'
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'filters': ['thread_control'],
            },

    """

    CONTEXT = ThreadContexts(counters='silenced')

    # we use this ordered-dict to track which thread is currently 'solo-ed'
    # we populate it with some initial values to make the 'filter' method
    # implementation more convenient
    SELECTED = OrderedDict()
    IDX_GEN = count()
    LOCK = threading.RLock()

    @classmethod
    @contextmanager
    def solo(cls):
        try:
            with cls.LOCK:
                idx = next(cls.IDX_GEN)
                cls.SELECTED[idx] = threading.current_thread()
            yield
        finally:
            cls.SELECTED.pop(idx)

    def filter(self, record):
        selected = False
        while selected is False:
            idx = next(reversed(self.SELECTED), None)
            if idx is None:
                selected = None
                break
            selected = self.SELECTED.get(idx, False)

        if selected:
            return selected == threading.current_thread()

        if self.CONTEXT.silenced is False:
            return False
        if self.CONTEXT.silenced is True:
            return True
        return record.levelno <= logging._nameToLevel.get(self.CONTEXT.silenced, self.CONTEXT.silenced)


class ColorizingFormatter(logging.Formatter):

    def formatMessage(self, record):
        if not hasattr(record, "levelcolor"):
            record.levelcolor = get_level_color(record.levelno)
        msg = super().formatMessage(record)
        return colorize(msg) if G.COLORING else uncolored(msg)


class ConsoleFormatter(ColorizingFormatter):

    def formatMessage(self, record):
        msg = super().formatMessage(record)
        if G.IS_A_TTY:
            msg = "\n".join('\r{0}{1}'.format(line, CLEAR_EOL) for line in msg.splitlines())
        return msg


try:
    import yaml
except ImportError:
    pass
else:
    try:
        from yaml import CDumper as Dumper
    except ImportError:
        from yaml import Dumper

    class YAMLFormatter(logging.Formatter):

        def __init__(self, **kw):
            self.dumper_params = kw

        def format(self, record):
            return yaml.dump(vars(record), Dumper=Dumper) + "\n---\n"

#=====================#=====================#=====================#
# This monkey-patch tricks logging's findCaller into skipping over
# this module when looking for the caller of a logger.log function
class _SrcFiles:
    _srcfiles = {logging._srcfile, __file__}
    def __eq__(self, fname):
        return fname in self.__class__._srcfiles
logging._srcfile = _SrcFiles()
#=====================#=====================#=====================#


_root = __file__[:__file__.find(os.sep.join(__name__.split(".")))]


def _trim(pathname, modname, cache={}):
    try:
        return cache[(pathname, modname)]
    except KeyError:
        pass

    elems = pathname.replace(_root, "").strip(".").split(os.sep)[:-1]
    if modname != "__init__":
        elems.append(modname)

    ret = cache[(pathname, modname)] = filter(None, elems)
    return ret


# =================================================================

def patched_makeRecord(self, name, level, fn, lno, msg, args, exc_info, func=None, extra=None, sinfo=None):
    decoration = G.graphics.INDENT_SEGMENT

    rv = self._makeRecord(name, level, fn, lno, msg, args, exc_info, func=func, sinfo=sinfo)
    if extra is not None:
        decoration = extra.pop('decoration', decoration)
        for key in extra:
            if (key in ["message", "asctime"]) or (key in rv.__dict__):
                raise KeyError("Attempt to overwrite %r in LogRecord" % key)
            rv.__dict__[key] = extra[key]

    contexts = THREAD_LOGGING_CONTEXT.context
    extra = THREAD_LOGGING_CONTEXT.flatten()
    extra['context'] = "[%s]" % ";".join(contexts) if contexts else ""
    rv.__dict__.update(dict(extra, **rv.__dict__))

    indents = chain(repeat(G.graphics.INDENT_SEGMENT, rv.indentation), repeat(decoration, 1))
    rv.decoration = "".join(color(segment) for color, segment in zip(cycle(G.INDENT_COLORS), indents))
    return rv
