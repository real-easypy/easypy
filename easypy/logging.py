# encoding: utf-8
from __future__ import absolute_import

import logging
import os
import random
import sys
import threading
import time
import traceback
from contextlib import contextmanager, ExitStack
from copy import deepcopy
from functools import wraps, partial
from itertools import cycle, chain, repeat

from easypy.bunch import Bunch
from easypy.colors import colorize_by_patterns as C, uncolorize
from easypy.humanize import compact as _compact
from easypy.timing import timing as timing_context, Timer

logging.INFO1 = logging.INFO+1
logging.addLevelName(logging.INFO1, "INFO1")


BASE_INDENTATION = 0
CLEAR_EOL = '\x1b[0K'
IS_A_TTY = sys.stdout.isatty()

LOG_WITH_COLOR=None
def compact(msg):
    raise NotImplementedError()

def set_width(TERM_WIDTH):
    if TERM_WIDTH in (True, None):
        TERM_WIDTH, _ = os.get_terminal_size()
        TERM_WIDTH = max(TERM_WIDTH, 120)
    if TERM_WIDTH:
        compact = lambda line: _compact(line, TERM_WIDTH-5)
    else:
        TERM_WIDTH = 0
        compact = lambda line: line
    globals().update(locals())


def set_coloring(enabled):
    global LOG_WITH_COLOR
    LOG_WITH_COLOR = enabled
    if enabled:
        from easypy.colors import RED, GREEN, BLUE, WHITE, DARK_GRAY
    else:
        RED = GREEN = BLUE = WHITE = DARK_GRAY = lambda txt, *_, **__: txt
    globals().update(locals())


# All this expected to be set by set_graphics call
LINE = None
DOUBLE_LINE = None
INDENT_SEGMENT = None
INDENT_OPEN = None
INDENT_CLOSE = None
INDENT_EXCEPTION = None

def set_graphics(GRAPHICAL):
    if GRAPHICAL:
        LINE = "─"
        DOUBLE_LINE = "═"
        INDENT_SEGMENT   = "  │ "
        INDENT_OPEN      = "  ├───┮ "
        INDENT_CLOSE         = "  ╰╼"
        INDENT_EXCEPTION     = "  ╘═"
    else:
        LINE = "-"
        DOUBLE_LINE = "="
        INDENT_SEGMENT   = "..| "
        INDENT_OPEN      = "..|---+ "
        INDENT_CLOSE         = "  '-"
        INDENT_EXCEPTION     = "  '="
    globals().update(locals())

set_width(IS_A_TTY)
set_coloring(IS_A_TTY or os.environ.get('TERM_COLOR_SUPPORT'))
set_graphics(IS_A_TTY or os.environ.get('TERM_COLOR_SUPPORT'))


LEVEL_COLORS = {
    logging.DEBUG:   "DARK_GRAY",
    logging.INFO:    "GRAY",
    logging.WARNING: "YELLOW",
    logging.ERROR:   "RED"
    }


def get_level_color(level):
    try:
        return LEVEL_COLORS[level]
    except KeyError:
        sorted_colors = sorted(LEVEL_COLORS.items(), reverse=True)
        for clevel, color in sorted_colors:
            if level > clevel:
                break
        LEVEL_COLORS[level] = color
        return color

INDENT_COLORS = [
    ("DARK_%s<<{}>>" % color.upper()).format
    for color in "GREEN BLUE MAGENTA CYAN YELLOW".split()]
random.shuffle(INDENT_COLORS)


class LogLevelClamp(logging.Filterer):
    def __init__(self, level=logging.DEBUG):
        self.level = level
        self.name = logging.getLevelName(level)
    def filter(self, record):
        if record.levelno > self.level:
            record.levelname, record.levelno = self.name, self.level
        return True


def get_console_handler():
    try:
        return logging._handlers['console']
    except KeyError:
        for handler in logging.root.handlers:
            if not isinstance(handler, logging.StreamHandler):
                continue
            return handler


class ConsoleFormatter(logging.Formatter):
    def formatMessage(self, record):
        if not hasattr(record, "levelcolor"):
            record.levelcolor = get_level_color(record.levelno)
        msg = super().formatMessage(record)
        if IS_A_TTY:
            msg = '\r' + msg + CLEAR_EOL
        msg = C(msg, not LOG_WITH_COLOR)
        return msg


class ContextHandler(logging.NullHandler):
    CONTEXT_DATA = {}
    DEFAULTS = {}

    def __init__(self, defaults={}):
        self.DEFAULTS.update(defaults)
        super().__init__()

    @classmethod
    def _get_context_data(cls, thread_uuid=None, combined=False):
        from easypy.threadtree import get_thread_uuid, get_parent_uuid

        if not thread_uuid:
            thread_uuid = get_thread_uuid()

        ctx = cls.CONTEXT_DATA.setdefault(thread_uuid, [])
        if not combined:
            return ctx

        parent_uuid = get_parent_uuid(thread_uuid)
        if parent_uuid:
            parent_ctx = cls._get_context_data(parent_uuid, combined=True)
            ctx = deepcopy(parent_ctx) + ctx
        return ctx

    @contextmanager
    def context(self, kw):
        for v in kw.values():
            assert (v is None) or (isinstance(v, (str, int, float))), "Can't use %r as context vars" % v
        ctx = self._get_context_data()
        ctx.append(Bunch(kw))
        try:
            yield
        except Exception as exc:
            context = get_current_context()
            if context and not getattr(exc, "context", None):
                try:
                    exc.context = context
                except:
                    logging.warning("could not attach context to exception")
            raise
        finally:
            ctx.pop(-1)

    def flatten(self, thread_uuid=None):
        stack = self._get_context_data(thread_uuid=thread_uuid, combined=True)
        contexts = []
        indentation = BASE_INDENTATION
        extra = dict(self.DEFAULTS)
        for ctx in stack:
            indentation += ctx.get("indentation", 0)
            contexts.append(ctx.get('context', None))
            extra.update(ctx)
        contexts = ";".join(str(s) for s in contexts if s)
        extra.update(
            context=("[%s] " % contexts) if contexts else "",
            indentation=indentation,
        )
        return extra

    def handle(self, record):
        extra = self.flatten()
        record.__dict__.update(dict(extra, **record.__dict__))
        drawing = getattr(record, "drawing", INDENT_SEGMENT)
        indents = chain(repeat(INDENT_SEGMENT, record.indentation), repeat(drawing, 1))
        record.drawing = "".join(color(segment) for color, segment in zip(cycle(INDENT_COLORS), indents))


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


# It essentially behaves as a singleton
CONTEXT_HANDLER = ContextHandler()
def get_current_context():
    return {k: v for k, v in CONTEXT_HANDLER.flatten().items()
            if v and k not in {"indentation"}}


def get_indentation():
    return CONTEXT_HANDLER.flatten().get("indentation", 0)


def _progress():
    from random import randint
    while True:
        yield chr(randint(0x2800, 0x28FF))


class ProgressBar:

    WAITING = "▅▇▆▃  ▆▇▆▅▃_       " #
    #PROGRESSING = "⣾⣽⣻⢿⡿⣟⣯⣷" #"◴◷◶◵◐◓◑◒"
    SPF = 1.0/15

    def __init__(self):
        self._event = threading.Event()
        self._thread = None
        self._lock = threading.RLock()
        self._depth = 0

    def loop(self):
        wait_seq = cycle(self.WAITING)
        prog_seq = _progress()
        wait_symb, progress_symb = map(next, (wait_seq, prog_seq))
        last_time = hanging = 0
        while True:
            progressed = self._event.wait(self.SPF)
            if self._stop:
                break
            now = time.time()

            if now - last_time >= self.SPF:
                wait_symb = next(wait_seq)
                last_time = now

            if progressed:
                progress_symb = next(prog_seq)
                hanging = 0
            else:
                hanging +=1

            anim = WHITE(wait_symb+progress_symb)

            elapsed = self._timer.elapsed.render(precision=0).rjust(8)
            if hanging >= (5*10*60):  # ~5 minutes with no log messages
                elapsed = RED(elapsed)
            else:
                elapsed = BLUE(elapsed)

            line = elapsed + self._last_line.rstrip()
            line = line.replace("__PB__", anim)
            print("\r" + line, end=CLEAR_EOL+"\r", flush=True)
            self._event.clear()
        print("\rDone waiting.", end=CLEAR_EOL+"\r", flush=True)

    def progress(self, record):
        if not self._thread:
            return
        if record.levelno >= logging.DEBUG:
            record.drawing = "__PB__" + record.drawing[2:]
            self._last_line = compact(uncolorize(self._format(record).split("\n")[0]).strip()[8:])
        self._event.set()

    def set_message(self, msg):
        msg = msg.replace("|..|", "|__PB__"+INDENT_SEGMENT[3])
        self._last_line = "|" + compact(msg)
        self._event.set()

    @contextmanager
    def __call__(self):
        if not GRAPHICAL:
            yield self
            return

        handler = get_console_handler()
        with self._lock:
            self._depth += 1
            if self._depth == 1:
                self.set_message("Waiting...")
                self._stop = False
                self._timer = Timer()
                self._format = handler.formatter.format if handler else lambda record: record.getMessage()
                self._thread = threading.Thread(target=self.loop, name="ProgressBar", daemon=True)
                self._thread.start()
        try:
            yield self
        finally:
            with self._lock:
                self._depth -= 1
                if self._depth <= 0:
                    self._stop = True
                    self._event.set()
                    self._thread.join()
                    self._thread = None


class ProgressHandler(logging.NullHandler):
    def handle(self, record):
        PROGRESS_BAR.progress(record)


PROGRESS_BAR = ProgressBar()


class ContextLoggerMixin(object):

    _debuggifier = LogLevelClamp()

    @contextmanager
    def context(self, context=None, indent=False, progress_bar=False, **kw):
        if context:
            kw['context'] = context
        with ExitStack() as stack:
            stack.enter_context(CONTEXT_HANDLER.context(kw))
            timing = kw.pop("timing", True)
            if indent:
                header = indent if isinstance(indent, str) else ("[%s]" % context)
                stack.enter_context(self.indented(header=header, timing=timing))
            if progress_bar:
                stack.enter_context(self.progress_bar())
            yield

    @contextmanager
    def suppressed(self, threshold=logging.ERROR):
        handler = get_console_handler()
        orig_level = handler.level
        handler.setLevel(threshold)
        try:
            yield
        finally:
            handler.setLevel(orig_level)

    @contextmanager
    def indented(self, header=None, *args, level=logging.INFO1, timing=True, footer=True):
        header = compact((header % args) if header else "")
        self._log(level, "WHITE@{%s}@" % header, (), extra=dict(drawing=INDENT_OPEN))
        with ExitStack() as stack:
            stack.enter_context(CONTEXT_HANDLER.context(dict(indentation=True)))

            get_duration = lambda: ""
            if timing:
                timer = stack.enter_context(timing_context())
                get_duration = lambda: " in DARK_MAGENTA<<{:text}>>".format(timer.duration)

            def footer_log(color, title, drawing):
                if footer:
                    self._log(level, "%s@{%s}@%s (%s)", (color, title, get_duration(), header), extra=dict(drawing=drawing))
                else:
                    self._log(level, "", (), extra=dict(drawing=drawing))

            try:
                yield
            except KeyboardInterrupt:
                footer_log("CYAN", "ABORTED", INDENT_EXCEPTION)
                raise
            except:
                footer_log("RED", "FAILED", INDENT_EXCEPTION)
                raise
            else:
                footer_log("DARK_GRAY", "DONE", INDENT_CLOSE)

    def error_box(self, *exc, extra=None):
        if len(exc)==1:
            exc, = exc
            typ = type(exc)
            tb = None
        else:
            typ, exc, tb = exc
        header = "%s.%s" % (typ.__module__, typ.__name__)
        self.error("YELLOW@{%s}@ RED@{%s}@", header, LINE*(80-len(header)-1), extra=dict(drawing=RED(INDENT_OPEN)))
        with CONTEXT_HANDLER.context(dict(indentation=True, drawing=RED(INDENT_SEGMENT))):
            if hasattr(exc, "render"):
                exc_text = exc.render()
            elif tb:
                fmt = "DARK_GRAY@{{{}}}@"
                full_traceback = "".join(traceback.format_exception(typ, exc, tb))
                exc_text = "\n".join(map(fmt.format, full_traceback.splitlines()))
            else:
                exc_text = str(exc)
            for line in exc_text.splitlines():
                self.error(line)
            if extra:
                for line in extra.splitlines():
                    self.error(line)
            self.error("RED@{%s}@", DOUBLE_LINE*80, extra=dict(drawing=RED(INDENT_EXCEPTION)))

    _progressing = False
    @contextmanager
    def progress_bar(self):
        if not GRAPHICAL:
            with PROGRESS_BAR() as pb:
                yield pb
                return

        with ExitStack() as stack:
            if not self.__class__._progressing:
                self.addFilter(self._debuggifier)
                stack.callback(self.removeFilter, self._debuggifier)
                stack.enter_context(PROGRESS_BAR())
                self.__class__._progressing = True
                stack.callback(setattr, self.__class__, "_progressing", False)
            yield PROGRESS_BAR

    def silent_exception(self, message, *args, **kwargs):
        "like exception(), only emits the traceback in debug level"
        self.error(message, *args, **kwargs)
        self.debug('Traceback:', exc_info=True)

    def __rand__(self, cmd):
        return cmd & self.pipe(logging.INFO, logging.INFO)

    def pipe(self, err_level=logging.DEBUG, out_level=logging.INFO, prefix=None, line_timeout=10 * 60, **kw):
        class LogPipe(object):
            def __rand__(_, cmd):
                for out, err in cmd.popen().iter_lines(line_timeout=line_timeout, **kw):
                    for level, line in [(out_level, out), (err_level, err)]:
                        if not line:
                            continue
                        for l in line.splitlines():
                            if prefix:
                                l = "%s: %s" % (prefix, l)
                            self.log(level, l)
        return LogPipe()

    def pipe_info(self, prefix=None, **kw):
        return self.pipe(logging.INFO, logging.INFO, prefix=prefix, **kw)

    def pipe_debug(self, prefix=None, **kw):
        return self.pipe(logging.DEBUG, logging.DEBUG, prefix=prefix, **kw)

    def info1(self, *args, **kwargs):
        return self.log(logging.INFO1, *args, **kwargs)

    def announced_vars(self, header='With locals:', *args, **kwargs):
        "Announces the variables declared in the context"
        import inspect
        frame = inspect.currentframe().f_back

        # `@contextmanager` annotates an internal `cm` function instead of the
        # `announced_vars` method so that `inspect.currentframe().f_back` will
        # point to the frame that uses `announced_vars`. If we decoraed
        # `announced_vars` with `@contextmanager`, we'd have to depend on
        # implementation details of `@contextmanager` - currently
        # `inspect.currentframe().f_back.f_back` would have worked, but we have
        # no guarantee that it'll remain like this forever.
        @contextmanager
        def cm():
            old_local_names = set(frame.f_locals.keys())
            yield
            new_locals = frame.f_locals
            with ExitStack() as stack:
                if header:
                    stack.enter_context(self.indented(header, *args, footer=False, **kwargs))
                # Traverse co_varnames to retain order
                for name in frame.f_code.co_varnames:
                    if name not in old_local_names and name in new_locals:
                        self.info('%s = %s', name, new_locals[name])

                # Print the names we somehow missed(because they weren't in co_varnames - it can happen!)
                for name in (new_locals.keys() - old_local_names - set(frame.f_code.co_varnames)):
                    self.info('%s = %s', name, new_locals[name])

        return cm()


class HeartbeatHandler(logging.Handler):
    "Heartbeat notifications based on the application's logging activity"

    def __init__(self, beat_func, min_interval=1, **kw):
        """
        @param beat_func: calls this function when a heartbeat is due
        @param min_interval: minimum time interval between heartbeats
        """
        super(HeartbeatHandler, self).__init__(**kw)
        self.min_interval = min_interval
        self.last_beat = 0
        self.beat = beat_func
        self._emitting = False

    def emit(self, record):
        if self._emitting:
            # prevent reenterance
            return

        try:
            self._emitting = True
            if (record.created - self.last_beat) > self.min_interval:
                try:
                    log_message = self.format(record)
                except:
                    log_message = "Log record formatting error (%s:#%s)" % (record.filename, record.lineno)
                self.beat(log_message=log_message, heartbeat=record.created)
                self.last_beat = record.created
        finally:
            self._emitting = False


def log_context(method=None, **ctx):
    if not method:
        return partial(log_context, **ctx)
    @wraps(method)
    def inner(*args, **kwargs):
        context = {k: fmt.format(*args, **kwargs) for k, fmt in ctx.items()}
        with CONTEXT_HANDLER.context(context):
            return method(*args, **kwargs)
    return inner


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
