# encoding: utf-8
from __future__ import absolute_import

import os
import logging
import random
import sys
import traceback
from contextlib import ExitStack
from functools import wraps, partial
from itertools import chain
from unittest.mock import MagicMock


class Graphics:

    class Graphical:
        LINE = "─"
        DOUBLE_LINE = "═"
        INDENT_SEGMENT   = "  │ "      # noqa
        INDENT_OPEN      = "  ├───┮ "  # noqa
        INDENT_CLOSE         = "  ╰╼"  # noqa
        INDENT_EXCEPTION     = "  ╘═"  # noqa

    class ASCII:
        LINE = "-"
        DOUBLE_LINE = "="
        INDENT_SEGMENT   = "..| "      # noqa
        INDENT_OPEN      = "..|---+ "  # noqa
        INDENT_CLOSE         = "  '-"  # noqa
        INDENT_EXCEPTION     = "  '="  # noqa


class G:
    """
    This class is not to be instantiated, or used outside of easypy.
    It is simply a namespace, container for lazily determined settings for this logging module.
    'G' stands for Global. It is one letter since it is used frequently in the code, and we don't want our lines too long.
    """

    initialized = False

    IS_A_TTY = None
    GRAPHICAL = None
    COLORING = None
    GRPH = Graphics.ASCII  # graphical elements for generating indentations in the log
    NOTICE = None  # the INFO+1 log-level; in 'logbook' there's a "NOTICE" log-level - we borrow their terminology
    TRACE = None   # borrowing from logbook's TRACE log evel
    LEVEL_COLORS = None


from easypy.aliasing import aliases


@aliases("logger")
class DeferredEasypyLogger():
    # this is a temporary mock-logger that can handle logging calls within easypy,
    # until logging is initialized, if it ever is.
    # When logging is initialized, an appropriate logger is placed on any previously
    # instantiated DeferredEasypyLogger object. New instances will create the appropriate
    # logger themselves
    logger = MagicMock(name="mocked-logger")
    _pending = []

    def __init__(self, name):
        self.name = name
        if G.initialized:
            self.logger = _get_logger(name)
        else:
            self._pending.append(self)


from easypy.threadtree import ThreadContexts
from easypy.contexts import contextmanager
from easypy.tokens import if_auto, AUTO


def get_level_color(level):
    try:
        return G.LEVEL_COLORS[level]
    except KeyError:
        sorted_colors = sorted(G.LEVEL_COLORS.items(), reverse=True)
        for clevel, color in sorted_colors:
            if level > clevel:
                break
        G.LEVEL_COLORS[level] = color
        return color


THREAD_LOGGING_CONTEXT = ThreadContexts(counters="indentation", stacks="context")
get_current_context = THREAD_LOGGING_CONTEXT.flatten


def get_indentation() -> int:
    """
    Return the current logging indentation
    """
    return THREAD_LOGGING_CONTEXT.indentation


class AbortedException(BaseException):
    """ Aborted base class

    Exceptions that inherit from this class will show as ABORTED in logger.indented
    """


class ContextableLoggerMixin(object):
    """
    A mixin class that provides easypy's logging functionality via the built-in logging's Logger objects:
        - context and indentation
        - per-thread logging supression and soloing
        - progress-bar support
        - and more...
    """

    @contextmanager
    def context(self, context=None, indent=False, progress_bar=False, **kw):
        if context:
            kw['context'] = context
        with ExitStack() as stack:
            stack.enter_context(THREAD_LOGGING_CONTEXT(kw))
            timing = kw.pop("timing", True)
            if indent:
                header = indent if isinstance(indent, str) else ("[%s]" % context)
                stack.enter_context(self.indented(header=header, timing=timing))
            if progress_bar:
                stack.enter_context(self.progress_bar())
            yield

    def suppressed(self, level=True):
        """
        Context manager - Supress all logging to the console from the calling thread
        """
        return ThreadControl.CONTEXT(silenced=level)

    def solo(self):
        """
        Context manager - Allow logging to the console from the calling thread only
        """
        return ThreadControl.solo()

    @contextmanager
    def indented(self, header=None, *args, level=AUTO, timing=True, footer=True):
        from easypy.timing import timing as timing_context

        level = if_auto(level, G.NOTICE)
        header = (header % args) if header else ""
        self.log(level, "WHITE@[%s]@" % header, extra=dict(decoration=G.graphics.INDENT_OPEN))
        with ExitStack() as stack:
            stack.enter_context(THREAD_LOGGING_CONTEXT(indentation=1))

            get_duration = lambda: ""
            if timing:
                timer = stack.enter_context(timing_context())
                get_duration = lambda: " in DARK_MAGENTA<<{:text}>>".format(timer.duration)

            def footer_log(color, title, decoration):
                if footer:
                    msg = "%s@[%s]@%s (%s)" % (color, title, get_duration(), header)
                    self.log(level, msg, extra=dict(decoration=decoration))
                else:
                    self.log(level, "", extra=dict(decoration=decoration))

            try:
                yield
            except (KeyboardInterrupt, AbortedException):
                footer_log("CYAN", "ABORTED", G.graphics.INDENT_EXCEPTION)
                raise
            except GeneratorExit:
                footer_log("DARK_GRAY", "DONE", G.graphics.INDENT_CLOSE)
            except:
                footer_log("RED", "FAILED", G.graphics.INDENT_EXCEPTION)
                raise
            else:
                footer_log("DARK_GRAY", "DONE", G.graphics.INDENT_CLOSE)

    def error_box(self, *exc, extra=None):
        """
        Generates a distinct red graphical box with exception information to the log, at 'ERROR' log-level.
        """
        if len(exc) == 1:
            exc, = exc
            typ = type(exc)
            tb = None
        else:
            typ, exc, tb = exc

        header = "{}.{}".format(typ.__module__, typ.__name__)
        self.error(
            "YELLOW@{%s}@ RED@{%s}@", header, G.graphics.LINE * (80 - len(header) - 1),
            extra=dict(decoration=G.RED(G.graphics.INDENT_OPEN)))

        with THREAD_LOGGING_CONTEXT(indentation=1, decoration=G.RED(G.graphics.INDENT_SEGMENT)):
            if hasattr(exc, "render") and callable(exc.render):
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
            self.error("RED@{%s}@", G.DOUBLE_LINE*80, extra=dict(decoration=G.RED(G.graphics.INDENT_EXCEPTION)))

    def silent_exception(self, message, *args, **kwargs):
        """
        Like ``exception()``, only emits the traceback in debug level
        """
        self.error(message, *args, **kwargs)
        self.debug('Traceback:', exc_info=True)

    def trace(self, *args, **kwargs):
        self.log(G.TRACE, *args, **kwargs)

    def notice(self, *args, **kwargs):
        """
        Log at 'NOTICE' log level, which is 'INFO1' in logging, and 'NOTICE' in logbook
        """
        return self.log(G.NOTICE, *args, **kwargs)

    info1 = notice  # for backwards compatibility

    def announced_vars(self, header='With locals:', *args, **kwargs):
        """
        Announces the variables declared in the context
        """
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

    def traced(self, func=None, *, with_params=False, with_rv=False):
        if not func:
            return partial(self.traced, with_params=with_params, with_rv=with_rv)

        from easypy.timing import timing as timing_context
        fullname = f"{func.__module__}:{func.__name__}"

        @wraps(func)
        def inner(*args, **kwargs):
            if with_params:
                params = ", ".join(chain(map(str, args), (f"{k}={v!r}" for k, v in kwargs.items())))
            else:
                params = "..."
            self.trace(f">> DARK_CYAN<<{fullname}>>({params})")
            with timing_context() as timer:
                try:
                    ret = func(*args, **kwargs)
                except BaseException as e:
                    self.trace(
                        f"!! DARK_RED<<{fullname}>> --X ({type(e)}) (MAGENTA<<{timer.elapsed}>>)")
                    raise
            rv = repr(ret) if with_rv else "" if ret is None else f"({type(ret)})"
            self.trace(
                f"<< DARK_CYAN<<{fullname}>> --> {rv} (MAGENTA<<{timer.elapsed}>>)")
            return ret

        return inner


def log_context(method=None, **ctx):
    if not method:
        return partial(log_context, **ctx)

    @wraps(method)
    def inner(*args, **kwargs):
        context = {k: fmt.format(*args, **kwargs) for k, fmt in ctx.items()}
        with THREAD_LOGGING_CONTEXT(context):
            return method(*args, **kwargs)
    return inner


# =================================================================

def initialize(*, graphical=AUTO, coloring=AUTO, indentation=0, context={}, patch=False, framework="logging"):
    """
    Initialize easypy's logging module.
    Also injects easypy's ContextableLoggerMixin into the builtin logging module.

    :param graphical: Whether to use unicode or ascii for graphical elements
    :param coloring: Whether to use color in logging
    :param indentation: The initial indentation level
    :param context: The initial logging context attributes
    """

    if G.initialized:
        logging.warning("%s is already initialized", __name__)
        return

    # our gevent patching prevents us from safely importing this function
    # from the parent 'easypy' module. workaround was to de-DRY it... (clone it)
    def yesno_to_bool(s):
        s = s.lower()
        if s not in ("yes", "no", "true", "false", "1", "0"):
            raise ValueError("Unrecognized boolean value: %r" % (s,))
        return s in ("yes", "true", "1")

    G.IS_A_TTY = sys.stdout.isatty()

    # =====================
    # Graphics initialization

    if graphical is AUTO:
        graphical = os.getenv('EASYPY_AUTO_GRAPHICAL_LOGGING', '')
        if graphical:
            graphical = yesno_to_bool(graphical)
        else:
            graphical = G.IS_A_TTY

    G.GRAPHICAL = graphical, G.IS_A_TTY
    G.graphics = Graphics.Graphical if G.GRAPHICAL else Graphics.ASCII

    # =====================
    # Coloring indentation

    if coloring is AUTO:
        coloring = os.getenv('EASYPY_AUTO_COLORED_LOGGING', '')
        if coloring:
            coloring = yesno_to_bool(coloring)
        else:
            coloring = G.IS_A_TTY

    G.COLORING = coloring
    if G.COLORING:
        from easypy.colors import RED, GREEN, BLUE, WHITE, DARK_GRAY
        G.INDENT_COLORS = [
            ("DARK_%s<<{}>>" % color.upper()).format
            for color in "GREEN BLUE MAGENTA CYAN YELLOW".split()]
        random.shuffle(G.INDENT_COLORS)
    else:
        RED = GREEN = BLUE = WHITE = DARK_GRAY = lambda txt, *_, **__: txt
        G.INDENT_COLORS = [lambda s: s]

    G.RED = RED
    G.GREEN = GREEN
    G.BLUE = BLUE
    G.WHITE = WHITE
    G.DARK_GRAY = DARK_GRAY

    # =====================
    # Context

    G._ctx = ExitStack()
    G._ctx.enter_context(THREAD_LOGGING_CONTEXT(indentation=indentation, **context))

    # =====================
    # Mixin injection
    from .heartbeats import HeartbeatHandlerMixin
    global HeartbeatHandler, EasypyLogger, get_console_handler, ColorizingFormatter, ConsoleFormatter
    global ThreadControl
    global _get_logger

    if framework == "logging":
        from .progressbar import ProgressBarLoggerMixin
        from ._logging import get_console_handler, LEVEL_COLORS, patched_makeRecord, ColorizingFormatter, ConsoleFormatter
        from ._logging import ThreadControl
        G.LEVEL_COLORS = LEVEL_COLORS

        logging.INFO1 = logging.INFO + 1
        logging.addLevelName(logging.INFO1, "INFO1")
        G.NOTICE = logging.INFO1
        G.TRACE = logging.NOTSET

        class ContextLoggerMixin(ContextableLoggerMixin, ProgressBarLoggerMixin):
            # for backwards compatibility
            pass

        class EasypyLogger(logging.Logger, ContextableLoggerMixin):
            pass

        # _get_logger should be used internally in easypy to get an appropriate logger object
        _get_logger = logging.getLogger

        if patch:
            logging.Logger._makeRecord, logging.Logger.makeRecord = logging.Logger.makeRecord, patched_makeRecord
            logging.setLoggerClass(EasypyLogger)
            logging.Logger.manager.setLoggerClass(EasypyLogger)

        class HeartbeatHandler(logging.Handler, HeartbeatHandlerMixin):
            pass

    elif framework == "logbook":
        import logbook
        from ._logbook import ContextProcessor, ThreadControl, ConsoleHandlerMixin
        from ._logbook import get_console_handler, LEVEL_COLORS, ColorizingFormatter, ConsoleFormatter
        from ._logbook import LoggingToLogbookAdapter
        G.LEVEL_COLORS = LEVEL_COLORS
        G.NOTICE = logbook.NOTICE
        G.TRACE = logbook.TRACE

        class HeartbeatHandler(logbook.Handler, HeartbeatHandlerMixin):
            pass

        class EasypyLogger(logbook.Logger, ContextableLoggerMixin):
            pass

        class InternalEasypyLogger(LoggingToLogbookAdapter, EasypyLogger):
            pass

        # _get_logger should be used internally in easypy to get an appropriate logger object
        _get_logger = InternalEasypyLogger

        if patch:
            ContextProcessor().push_application()
            ThreadControl().push_application()
            logbook.StderrHandler.__bases__ = (ConsoleHandlerMixin,) + logbook.StderrHandler.__bases__

    else:
        raise NotImplementedError("No support for %s as a logging framework" % framework)

    for obj in DeferredEasypyLogger._pending:
        obj.logger = _get_logger(name=obj.name)
    DeferredEasypyLogger._pending.clear()

    G.initialized = framework
