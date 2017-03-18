from __future__ import absolute_import
import sys
import traceback
from time import time
from datetime import datetime
from contextlib import contextmanager
from textwrap import indent
from logging import getLogger
_logger = getLogger(__name__)


class PException(Exception):

    """An exception object that can accept kwargs as attributes"""

    def __init__(self, message="", *args, **params):
        if args or params:
            message = message.format(*args, **params)
        Exception.__init__(self, message)
        self.context = params.pop("context", None)
        self.traceback = params.pop("traceback", None)
        if self.traceback is True:
            self.traceback = traceback.format_exc()
        self.message = message
        self.timestamp = params.pop('timestamp', time())
        if 'tip' not in params:
            # sometimes it's on the class
            params['tip'] = getattr(self, 'tip', None)
        for k, v in params.items():
            setattr(self, k, v)
        self.params = params

    def __repr__(self):
        if self.params:
            kw = sorted("%s=%r" % (k, v) for k, v in self.params.items())
            return "%s(%r, %s)" % (self.__class__.__name__, self.message, ", ".join(kw))
        else:
            return "%s(%r)" % (self.__class__.__name__, self.message)

    def __str__(self):
        return self.render(traceback=False, color=False)

    def render(self, params=True, context=True, traceback=True, timestamp=True, color=True):
        text = ""

        if self.message:
            text += ("WHITE<<%s>>\n" % self.message)

        if params and self.params:
            tip = self.params.pop('tip', None)
            text += indent("".join(make_block(self.params)), " "*4)
            if tip:
                tip = tip.format(**self.params)
                text += indent("GREEN(BLUE)@{tip = %s}@\n" % tip, " "*4)
                self.params['tip'] = tip  # put it back in params, even though it might've been on the class

        if context and self.context:
            text += "Context:\n" + indent("".join(make_block(self.context)), " "*4)

        if timestamp and self.timestamp:
            ts = datetime.fromtimestamp(self.timestamp).isoformat()
            text += "Timestamp: MAGENTA<<%s>>\n" % ts

        if traceback and self.traceback:
            fmt = "DARK_GRAY@{{{}}}@"
            text += "\n".join(map(fmt.format, self.traceback.splitlines()))

        if not color:
            from easypy.colors import colorize_by_patterns
            text = colorize_by_patterns(text, no_color=True)

        return text

    @classmethod
    def make(cls, name):
        return type(name, (cls,), {})

    @classmethod
    @contextmanager
    def on_exception(cls, acceptable=Exception, **kwargs):
        try:
            yield
        except cls:
            # don't mess with exceptions of this type
            raise
        except acceptable as exc:
            exc_info = sys.exc_info()
            _logger.debug("'%s' raised; Raising as '%s'" % (type(exc), cls), exc_info=exc_info)
            raise cls(traceback=True, **kwargs) from None


def make_block(d):
    for k in sorted(d):
        if k.startswith("_"):
            continue
        v = d[k]
        if not isinstance(v, str):
            v = repr(v)
        dark = False
        if k.startswith("~"):
            k = k[1:]
            dark = True
        head = "%s = " % k
        block = indent(v, " "*len(head))
        block = head + block[len(head):]
        if dark:
            block = "DARK_GRAY@{%s}@" % block
        yield block + "\n"


class TException(PException):

    @property
    def template(self):
        raise NotImplementedError("Must implement template")

    def __init__(self, *args, **params):
        super(TException, self).__init__(self.template, *args, **params)

    @classmethod
    def make(cls, name, template):
        return type(name, (cls,), dict(template=template))


def convert_traceback_to_list(tb):
    # convert to list of dictionaries that contain file, line_no and function
    traceback_list = [dict(file=file, line_no=line_no, function=function)
                      for file, line_no, function, _ in traceback.extract_tb(tb)]
    return traceback_list
