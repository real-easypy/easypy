#!/usr/bin/env python
from __future__ import division, absolute_import, print_function, unicode_literals

import re
import functools
import sys


color_names = [
    'dark_gray',
    'red',
    'green',
    'yellow',
    'blue',
    'magenta',
    'cyan',
    'white',
]

COLORS = dict(
    (name, dict(color=color_num, light=True))
    for color_num, name in enumerate(color_names)
)

for orig in sorted(COLORS):
    COLORS['dark_' + orig] = dict(COLORS[orig], light=False)

for orig, dark in [('dark_gray', 'black'),
                   ('white', 'gray')]:
    COLORS[dark] = dict(COLORS[orig], light=False)

# TODO - tokenize an ascii-colored string
# list(re.finditer("\\x1b\[([\d;]+)m(.*?)\\x1b\[m", x))

MAGIC = "\x1b"
END_CODE = MAGIC + "[m"

_RE_ANSI_COLOR = re.compile(re.escape(MAGIC) + '.+?m')
_COLORIZER_RE_PATTERN = re.compile("([\w]+)(?:\((.*)\))?")
_COLOR_RE_PATTERN = re.compile(r"(?ms)([A-Z_]+(?:\([^\)]+\))?(?:(?:\<\<).*?(?:\>\>)|(?:\@\{).*?(?:\}\@)))")
_COLORING_RE_PATTERN = re.compile(r"(?ms)([A-Z_]+(?:\([^\)]+\))?)((?:\<\<.*?\>\>|\@\{.*?\}\@))")


def colorize(text, color='white', background=None, underline=False):
    if not isinstance(text, str):
        text = str(text)
    fmt = _get_colorize_formatter(color, background, underline)
    return fmt(text)

def uncolorize(text):
    return _RE_ANSI_COLOR.sub("", text)


class Colorized(str):

    class Token(str):

        def raw(self):
            return self

        def copy(self, text):
            return self.__class__(text)

        def __getslice__(self, start, stop):
            return self[start:stop:]

        def __getitem__(self, *args):
            return self.copy(str.__getitem__(self, *args))

        def __iter__(self):
            for c in str.__str__(self):
                yield self.copy(c)

    class ColoredToken(Token):

        def __new__(cls, text, colorizer_name):
            self = str.__new__(cls, text)
            self.__name = colorizer_name
            return self

        def __str__(self):
            return get_colorizer(self.__name)(str.__str__(self))

        def copy(self, text):
            return self.__class__(text, self.__name)

        def raw(self):
            return "%s<<%s>>" % (self.__name, str.__str__(self))

        def __repr__(self):
            return repr(self.raw())

    def __new__(cls, text):
        text = uncolorize(text)  # remove exiting colors
        self = str.__new__(cls, text)
        self.tokens = []
        for text in _COLOR_RE_PATTERN.split(text):
            match = _COLORING_RE_PATTERN.match(text)
            if match:
                stl = match.group(1).strip("_")
                text = match.group(2)[2:-2]
                for l in text.splitlines():
                    self.tokens.append(self.ColoredToken(l, stl))
                    self.tokens.append(self.Token("\n"))
                if not text.endswith("\n"):
                    del self.tokens[-1]
            else:
                self.tokens.append(self.Token(text))
        self.uncolored = "".join(str.__str__(token) for token in self.tokens)
        self.colored = "".join(str(token) for token in self.tokens)
        return self

    def raw(self):
        return str.__str__(self)

    def __str__(self):
        return self.colored

    def withuncolored(func):
        def inner(self, *args):
            return func(self.uncolored, *args)
        return inner

    __len__ = withuncolored(len)
    count = withuncolored(str.count)
    endswith = withuncolored(str.endswith)
    find = withuncolored(str.find)
    index = withuncolored(str.index)
    isalnum = withuncolored(str.isalnum)
    isalpha = withuncolored(str.isalpha)
    isdigit = withuncolored(str.isdigit)
    islower = withuncolored(str.islower)
    isspace = withuncolored(str.isspace)
    istitle = withuncolored(str.istitle)
    isupper = withuncolored(str.isupper)
    rfind = withuncolored(str.rfind)
    rindex = withuncolored(str.rindex)

    def withcolored(func):
        def inner(self, *args):
            return self.__class__("".join(t.copy(func(t, *args)).raw() for t in self.tokens if t))
        return inner

    #capitalize = withcolored(str.capitalize)
    expandtabs = withcolored(str.expandtabs)
    lower = withcolored(str.lower)
    replace = withcolored(str.replace)

    # decode = withcolored(str.decode)
    # encode = withcolored(str.encode)
    swapcase = withcolored(str.swapcase)
    title = withcolored(str.title)
    upper = withcolored(str.upper)

    def __getitem__(self, idx):
        if isinstance(idx, slice) and idx.step is None:
            start = idx.start or 0
            stop = idx.stop or len(self)
            cursor = 0
            tokens = []
            for token in self.tokens:
                tokens.append(token[max(0, start - cursor):stop - cursor])
                cursor += len(token)
                if cursor > stop:
                    break
            return self.__class__("".join(t.raw() for t in tokens if t))

        tokens = [c for token in self.tokens for c in token].__getitem__(idx)
        return self.__class__("".join(t.raw() for t in tokens if t))

    def __add__(self, other):
        return self.__class__("".join(map(str.__str__, (self, other))))

    def __mod__(self, other):
        return self.__class__(self.raw() % other)

    def format(self, *args, **kwargs):
        return self.__class__(self.raw().format(*args, **kwargs))

    def rjust(self, *args):
        padding = self.uncolored.rjust(*args)[:-len(self.uncolored)]
        return self.__class__(padding + self.raw())

    def ljust(self, *args):
        padding = self.uncolored.ljust(*args)[len(self.uncolored):]
        return self.__class__(self.raw() + padding)

    def center(self, *args):
        padded = self.uncolored.center(*args)
        return self.__class__(padded.replace(self.uncolored, self.raw()))

    def join(self, *args):
        return self.__class__(self.raw().join(*args))

    def _iter_parts(self, parts):
        last_cursor = 0
        for part in parts:
            pos = self.uncolored.find(part, last_cursor)
            yield self[pos:pos + len(part)]
            last_cursor = pos + len(part)

    def withiterparts(func):
        def inner(self, *args):
            return list(self._iter_parts(func(self.uncolored, *args)))
        return inner

    split = withiterparts(str.split)
    rsplit = withiterparts(str.rsplit)
    splitlines = withiterparts(str.splitlines)
    partition = withiterparts(str.partition)
    rpartition = withiterparts(str.rpartition)

    def withsingleiterparts(func):
        def inner(self, *args):
            return next(self._iter_parts([func(self.uncolored, *args)]))
        return inner

    strip = withsingleiterparts(str.strip)
    lstrip = withsingleiterparts(str.lstrip)
    rstrip = withsingleiterparts(str.rstrip)

    def zfill(self, *args):
        padding = self.uncolored.zfill(*args)[:-len(self.uncolored)]
        return self.__class__(padding + self.raw())

C = Colorized


def colored(string, *args):
    return str(C(string % args if args else string))


def colorize_by_patterns(text, no_color=False):
    if no_color:
        def _subfunc(match_obj):
            return match_obj.group(2)[2:-2]
    else:
        def _subfunc(match_obj):
            return get_colorizer(match_obj.group(1))(match_obj.group(2)[2:-2])

    text = _COLORING_RE_PATTERN.sub(_subfunc, text)
    if no_color:
        text = uncolorize(text)
    return text


COLORIZERS = {}


def _get_colorize_formatter(color='white', background=None, underline=False):
    key = color, background, underline

    try:
        return COLORIZERS[key]
    except KeyError:
        pass

    light = COLORS[color]['light']
    color = COLORS[color]['color']
    if not underline and not light:
        prefix_str = MAGIC + "[0"
    elif not underline or not light:
        prefix_str = MAGIC + "[%d" % (light and 1 or 4)
    else:
        prefix_str = MAGIC + "[4m" + MAGIC + "[1"
    if background is not None:
        background = COLORS[background]['color']
        color_str = "%d;%d" % (background + 40, color + 30)
    else:
        color_str = "%d" % (color + 30,)

    fmt = "%s;%sm{}%s" % (prefix_str, color_str, END_CODE)

    COLORIZERS[key] = fmt.format
    return fmt.format


def get_colorizer(name):
    try:
        return COLORIZERS[name.lower().strip("_")]
    except KeyError:
        pass

    color, background = (c and c.lower().strip("_") for c in _COLORIZER_RE_PATTERN.match(name).groups())
    if color not in COLORS:
        color = "white"
    if background not in COLORS:
        background = None
    return add_colorizer(name, color, background)


def add_colorizer(name, color="white", background=None):
    COLORIZERS[name.lower().strip("_")] = colorizer = _get_colorize_formatter(color=color, background=background)
    return colorizer


def init_colorizers(styles):
    for name, style in styles.items():
        add_colorizer(name, color=style['fg'], background=style['bg'])


globals().update((name.upper(), get_colorizer(name)) for name in COLORS)


if __name__ == '__main__':
    import fileinput
    for line in fileinput.input():
        print(colorize_by_patterns(line), end="", flush=True)
