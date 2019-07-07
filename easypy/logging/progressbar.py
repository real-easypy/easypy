# encoding: utf-8
from __future__ import absolute_import

import logging
import os
import threading
import time
from contextlib import ExitStack
from itertools import cycle

from easypy.colors import uncolored
from easypy.humanize import compact
from easypy.timing import Timer
from easypy.contexts import contextmanager
from easypy.misc import at_least
from easypy.logging import G

CLEAR_EOL = '\x1b[0K'


def _progress():
    from random import randint
    while True:
        yield chr(randint(0x2800, 0x28FF))


class ProgressBar:

    WAITING = "▅▇▆▃  ▆▇▆▅▃_       "
    # PROGRESSING = "⣾⣽⣻⢿⡿⣟⣯⣷" #"◴◷◶◵◐◓◑◒"
    SPF = 1.0 / 15

    def __init__(self):
        self._event = threading.Event()
        self._thread = None
        self._lock = threading.RLock()
        self._depth = 0
        self._term_width, _ = os.get_terminal_size() if G.IS_A_TTY else [0, 0]
        self._term_width = at_least(120, self._term_width)

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
                hanging += 1

            anim = G.WHITE(wait_symb + progress_symb)

            elapsed = self._timer.elapsed.render(precision=0).rjust(8)
            if hanging >= (5 * 10 * 60):  # ~5 minutes with no log messages
                elapsed = G.RED(elapsed)
            else:
                elapsed = G.BLUE(elapsed)

            line = elapsed + self._last_line.rstrip()
            line = line.replace("__PB__", anim)
            print("\r" + line, end=CLEAR_EOL + "\r", flush=True)
            self._event.clear()
        print("\rDone waiting.", end=CLEAR_EOL + "\r", flush=True)

    def progress(self, record):
        if not self._thread:
            return

        if record.levelno >= logging.DEBUG:
            record.decoration = "__PB__" + record.decoration[2:]
            txt = uncolored(self._format(record).split("\n")[0]).strip()[8:]
            self._last_line = compact(txt, self._term_width - 5)
        self._event.set()

    def set_message(self, msg):
        msg = msg.replace("|..|", "|__PB__" + G.graphics.INDENT_SEGMENT[3])
        self._last_line = "|" + compact(msg, self._term_width - 5)
        self._event.set()

    @contextmanager
    def __call__(self):
        if not G.GRAPHICAL:
            yield self
            return

        from . import get_console_handler
        handler = get_console_handler()

        if not isinstance(handler, logging.Handler):
            # not supported
            yield self
            return

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


class ProgressHandlerMixin():
    def handle(self, record):
        PROGRESS_BAR.progress(record)


PROGRESS_BAR = ProgressBar()


class ProgressBarLoggerMixin():

    _progressing = False
    @contextmanager
    def progress_bar(self):
        if not G.GRAPHICAL:
            with PROGRESS_BAR() as pb:
                yield pb
                return

        from . import LogLevelClamp
        debuggifier = LogLevelClamp(logger=self)

        with ExitStack() as stack:
            if not self.__class__._progressing:
                stack.enter_context(debuggifier)
                stack.enter_context(PROGRESS_BAR())
                self.__class__._progressing = True
                stack.callback(setattr, self.__class__, "_progressing", False)
            yield PROGRESS_BAR
