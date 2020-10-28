# encoding: utf-8
from __future__ import absolute_import

import logbook
import re
import sys
import threading
import logging
from itertools import cycle, chain, repeat, count
from collections import OrderedDict

from easypy.colors import colorize, uncolored
from easypy.threadtree import ThreadContexts
from easypy.contexts import contextmanager

from easypy.logging import G, get_level_color, THREAD_LOGGING_CONTEXT


CLEAR_EOL = '\x1b[0K'


LEVEL_COLORS = {
    logbook.DEBUG: "DARK_GRAY",
    logbook.INFO: "GRAY",
    logbook.WARNING: "YELLOW",
    logbook.ERROR: "RED",
    logbook.NOTICE: "WHITE",
}


class LogLevelClamp(logbook.Processor):
    """
    Log-records with a log-level that is too high are clamped down.
    Used internally by the ``ProgressBar``
    """

    def __init__(self, logger=None, level=logbook.DEBUG):
        self.level = level
        self.name = logbook.get_level_name(logbook.lookup_level(level))

    def process(self, record):
        if record.levelno > self.level:
            record.levelname, record.levelno = self.name, self.level
        return True

    def __enter__(self):
        self.push_application()

    def __exit__(self, *args):
        self.pop_application()


def get_console_handler():
    stack_manager = logbook.Handler.stack_manager
    stderr_handlers = (h for h in stack_manager.iter_context_objects() if isinstance(h, logbook.StderrHandler))
    return next(stderr_handlers, None)


RE_OLD_STRING_FORMATTING = re.compile(r'%(?:\((\w+)\))?([ \-+#0]*)(\d*\.?\d*)([diouxXeEfFgGcrs])')


def convert_string_template(string):
    index = -1

    def repl(matched):
        nonlocal index
        keyword, flags, width, convert = matched.groups()
        if keyword:
            return matched.group(0)  # don't change
        else:
            index += 1
            align = sign = zero = type = ''
            flags = set(flags)

            if '-' in flags:
                align = "<"

            if '+' in flags:
                sign = '+'
            elif ' ' in flags:
                sign = ' '

            if '0' in flags:
                zero = "0"

            if convert in 'asr':
                convert = "!" + convert
            elif convert in 'cdoxXeEfFgG':
                convert, type = '', convert
            elif convert == 'i':
                convert, type = '', 'd'

            return "{%(index)d%(convert)s:%(align)s%(sign)s%(zero)s%(width)s%(type)s}" % locals()

    return RE_OLD_STRING_FORMATTING.sub(repl, string)


class LoggingToLogbookAdapter():
    """
    Converts %-style to {}-style format strings
    Converts logging log-levels to logbook log-levels
    """

    @classmethod
    def _to_logbook_level(cls, level):
        if level >= logging.CRITICAL:
            return logbook.CRITICAL
        if level >= logging.ERROR:
            return logbook.ERROR
        if level >= logging.WARNING:
            return logbook.WARNING
        if level > logging.INFO:
            return logbook.NOTICE
        if level >= logging.INFO:
            return logbook.INFO
        if level >= logging.DEBUG:
            return logbook.DEBUG
        return logbook.NOTSET

    def _log(self, level, args, kwargs):
        level = self._to_logbook_level(level)
        fmt = args[0]
        fmt = convert_string_template(fmt)
        args = (fmt,) + args[1:]
        kwargs.update(frame_correction=kwargs.get('frame_correction', 0) + 2)
        return super()._log(level, args, kwargs)


class ThreadControl(logbook.Processor):
    """
    Used by ContextLoggerMixin .solo and .suppressed methods to control logging to console
    """

    CONTEXT = ThreadContexts(defaults=dict(silenced=False))

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

    def process(self, record):
        selected = False
        while selected is False:
            idx = next(reversed(self.SELECTED), None)
            if idx is None:
                selected = None
                break
            selected = self.SELECTED.get(idx, False)

        if selected:
            record.extra['silenced'] = selected != threading.current_thread()
        elif self.CONTEXT.silenced is False:
            pass
        elif self.CONTEXT.silenced is True:
            record.extra['silenced'] = True
        else:
            record.extra['silenced'] = record.level <= logbook.lookup_level(self.CONTEXT.silenced)


class ConsoleHandlerMixin():

    def should_handle(self, record):
        return not record.extra.get("silenced", False) and super().should_handle(record)


class ColorizingFormatter(logbook.StringFormatter):

    def format_record(self, record, handler):
        if "levelcolor" not in record.extra:
            record.extra["levelcolor"] = get_level_color(record.level)
        msg = super().format_record(record, handler)
        return colorize(msg) if G.COLORING else uncolored(msg)


class ConsoleFormatter(ColorizingFormatter):

    def format_record(self, record, handler):
        msg = super().format_record(record, handler)
        if not isinstance(handler, logbook.StreamHandler):
            pass
        elif handler.stream not in (sys.stdout, sys.stderr):
            pass
        elif not G.IS_A_TTY:
            pass
        else:
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

    class YAMLFormatter(logbook.StringFormatter):

        def __init__(self, **kw):
            self.dumper_params = kw

        def __call__(self, record, handler):
            return yaml.dump(vars(record), Dumper=Dumper) + "\n---\n"


class ContextProcessor(logbook.Processor):

    def process(self, record):
        decoration = G.graphics.INDENT_SEGMENT
        extra = record.extra
        if extra is not None:
            decoration = extra.pop('decoration', decoration)

        contexts = THREAD_LOGGING_CONTEXT.context
        extra = THREAD_LOGGING_CONTEXT.flatten()
        extra['context'] = "[%s]" % ";".join(contexts) if contexts else ""
        record.extra.update(extra)
        indentation = record.extra['indentation']

        indents = chain(repeat(G.graphics.INDENT_SEGMENT, indentation), repeat(decoration, 1))
        record.extra['decoration'] = "".join(color(segment) for color, segment in zip(cycle(G.INDENT_COLORS), indents))
        return record
