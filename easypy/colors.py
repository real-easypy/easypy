#!/usr/bin/env python

"""
This module allows rendering colored text in the console using a custom markup language::

    >> print(colorize("This will render as BLUE<<blue>>, and this will be YELLOW(BLUE)@{yellow-on-blue}@"))

"""

import re
import functools
from collections import namedtuple

ColorSpec = namedtuple("ColorSpec", "color light".split())


color_names = [
    'dark_gray',  # 0
    'red',        # 1
    'green',      # 2
    'yellow',     # 3
    'blue',       # 4
    'magenta',    # 5
    'cyan',       # 6
    'white',      # 7
]

COLORS = dict(
    (name, ColorSpec(color=color_num, light=True))
    for color_num, name in enumerate(color_names)
)

for orig in sorted(COLORS):
    COLORS['dark_' + orig] = ColorSpec(COLORS[orig].color, light=False)

for orig, dark in [('dark_gray', 'black'),
                   ('white', 'gray')]:
    COLORS[dark] = ColorSpec(COLORS[orig].color, light=False)


ANSI_BEGIN = "\x1b"
ANSI_END = ANSI_BEGIN + "[0m"


# this regex is used to split text into ANSI-colored/uncolored text blocks
RE_FIND_ANSI_COLORING = re.compile(r"({0}.*?{0}\[0m)".format(ANSI_BEGIN))

# this regex is used to tokenize an ANSI-colored string
RE_PARSE_ANSI_COLORING = re.compile(
    r"(?P<underline>{0}\[4m)"
    r"?{0}\[(?P<light>[01]\d*)"
    r"(?P<color>(?:;[1-9]\d*)*)m"
    r"(?P<text>.*?)"
    r"{0}\[0m".format(ANSI_BEGIN))


# this regex is used to split text into markup/non-markup text blocks
RE_FIND_COLOR_MARKUP = re.compile(
    r"(?ms)("
    r"[A-Z_]+(?:\([^\)]+\))?"
    r"(?:"
    r"(?:\<\<).*?(?:\>\>)|"
    r"(?:\@\{).*?(?:\}\@)|"
    r"(?:\@\[).*?(?:\]\@)"
    "))")

# this regex is used to parse the color markup into a foreground color, optional background, and the text itself.
# the text can be enclosed either by '<<..>>' or '@[...]@'
RE_PARSE_COLOR_MARKUP = re.compile(
    r"(?ms)"
    r"([A-Z_]+(?:\([^\)]+\))?)"  # group 0: the coloring
    r"(?:"
    r"\<\<(.*?)\>\>|"            # group 1: first trap for text <<...>>
    r"\@\{(.*?)\}\@|"            # group 2: second trap for text @{...}@
    r"\@\[(.*?)\]\@"             # group 3: second trap for text @[...]@
    ")")


class Colorized(str):
    """
    A string with coloring mark-up that retains string operation without interfering with the markup.
    For example::

        >> len(Colorized("RED<<red>>"))
        3

        >> Colorized("RED<<Red>>").lower()
        "RED<<red>>"
    """

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
            self.__colored = Colorizer.from_markup(self.__name)(text)
            return self

        def __str__(self):
            return self.__colored

        def copy(self, text):
            return self.__class__(text, self.__name)

        def raw(self):
            return "%s<<%s>>" % (self.__name, str.__str__(self))

        def __repr__(self):
            return repr(self.raw())

    def __new__(cls, text):
        text = uncolored(text, markup=False)  # remove exiting ANSI colors
        self = str.__new__(cls, text)
        self.tokens = []
        for part in RE_FIND_COLOR_MARKUP.split(text):
            if not part:
                continue
            match = RE_PARSE_COLOR_MARKUP.match(part)
            if match:
                stl, *parts = match.groups()
                stl = stl.strip("_")
                part = next(filter(None, parts))
                for l in part.splitlines():
                    self.tokens.append(self.ColoredToken(l, stl))
                    self.tokens.append(self.Token("\n"))
                if not part.endswith("\n"):
                    del self.tokens[-1]
            else:
                self.tokens.append(self.Token(part))
        self.uncolored = "".join(str.__str__(token) for token in self.tokens)
        self.colored = "".join(str(token) for token in self.tokens)
        return self

    def raw(self):
        return str.__str__(self)

    def len_delta(self):
        return len(self.colored) - len(self.uncolored)

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

    capitalize = withcolored(str.capitalize)
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

    def __radd__(self, other):
        return self.__class__("".join(map(str.__str__, (other, self))))

    def __mod__(self, other):
        return self.__class__(self.raw() % other)

    def __rmod__(self, other):
        return self.__class__(other % self.raw())

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

    @classmethod
    def from_ansi(cls, text):
        """
        Parse ANSI coloring and convert into Colorized text
        """
        parts = []
        for part in RE_FIND_ANSI_COLORING.split(text):
            if not part:
                continue
            match = RE_PARSE_ANSI_COLORING.match(part)
            if match:
                d = match.groupdict()
                fg = bg = None
                for color in d['color'].split(";"):
                    if not color:
                        continue
                    color = int(color)
                    if color >= 40:
                        bg = color_names[color - 40]
                    else:
                        fg = color_names[color - 30]

                color = fg.upper()
                if d['light'] in (None, "", "0"):
                    color = "DARK_{}".format(color)
                if bg:
                    color = "{}({})".format(color, bg.upper())
                part = d['text']
                if "<<" in part or ">>" in part:
                    sep = "@{", "}@"
                else:
                    sep = "<<", ">>"

                parts.append("{color}{sep[0]}{part}{sep[1]}".format(**locals()))
            else:
                parts.append(part)
        return cls("".join(parts))

    del withuncolored, withcolored


C = Colorized


def uncolored(text, ansi=True, markup=True):
    """
    Strip coloring markup and/or ANSI escape coloing from text
    """
    if ansi:
        text = re.sub(re.escape(ANSI_BEGIN) + '.+?m', "", text)
    if markup:
        text = RE_PARSE_COLOR_MARKUP.sub(lambda m: next(filter(None, m.groups()[1:])), text)
    return text


def colorize(text):
    """
    Colorize text according to markup
    """
    if not isinstance(text, str):
        text = str(text)

    def _subfunc(match_obj):
        colorizer = Colorizer.from_markup(match_obj.group(1))
        return colorizer(next(filter(None, match_obj.groups()[1:])))

    return RE_PARSE_COLOR_MARKUP.sub(_subfunc, text)


class Colorizer():
    """
    A callable for styling text::

        >> warning = Colorizer('yellow', 'blue', underline=True, name='warning')
        >> warning("this is a warning")
        '\\x1b[4m\\x1b[1;44;33mthis is a warning\\x1b[0m'
    """

    COLORIZERS = {}

    def __new__(cls, color='white', background=None, underline=False, name=None):
        if name:
            name = name.lower()
            try:
                return cls.COLORIZERS[name]
            except KeyError:
                pass

        key = color, background, underline

        try:
            ret = cls.COLORIZERS[key]
        except KeyError:
            ret = cls.COLORIZERS[key] = super().__new__(cls)

        if name:
            cls.COLORIZERS[name] = ret

        return ret

    def __init__(self, color='white', background=None, underline=False, name=None):
        self.color = color
        self.background = background
        self.underline = underline
        self.name = name.lower() if name else None

        color_spec = COLORS[color]
        light = color_spec.light
        color = color_spec.color
        if not underline and not light:
            prefix_str = ANSI_BEGIN + "[0"
        elif not underline or not light:
            prefix_str = ANSI_BEGIN + "[%d" % (1 if light else 4)
        else:
            prefix_str = ANSI_BEGIN + "[4m" + ANSI_BEGIN + "[1"
        if background is not None:
            background = COLORS[background].color
            color_str = "%d;%d" % (background + 40, color + 30)
        else:
            color_str = "%d" % (color + 30,)

        self.fmt = "%s;%sm{}%s" % (prefix_str, color_str, ANSI_END)

    def __repr__(self):
        if self.name:
            return "{0.__class__.__name__}('{0.name}')".format(self)
        else:
            spec = self.color
            if self.background:
                spec = "{}/{}".format(spec, self.background)
            if self.underline:
                spec += ", underlined"
            return "{0.__class__.__name__}({1})".format(self, spec)

    def __call__(self, text):
        return self.fmt.format(text)

    # this regex is used to parse the color spec in a color markup into a foreground color and an optional background color
    RE_PARSE_COLOR_SPEC = re.compile(r"([\w]+)(?:\((.*)\))?")

    @classmethod
    def from_markup(cls, markup):
        try:
            return cls.COLORIZERS[markup.lower()]
        except KeyError:
            pass
        color, background = (c and c.lower().strip("_") for c in cls.RE_PARSE_COLOR_SPEC.match(markup).groups())
        if color not in COLORS:
            color = "white"
        if background not in COLORS:
            background = None
        return cls(color=color, background=background, name=markup)


def register_colorizers(**styles):
    """
    Install named colorizers::

        >> register_colorizers(
            warning="yellow",               # yellow foreground
            critical=("yellow", "red"),     # yellow on red
            url=("white", "blue", True),    # white on blue with underline
            )
    """

    ret = {}
    for name, style in styles.items():
        if isinstance(style, str):
            fg = style
            bg = None
            underline = False
        elif isinstance(style, tuple):
            fg, bg, *more = style
            underline = more[0] if more else False
        else:
            raise ValueError("Invalid style: %r (expecting an str or a fg/bg tuple)", (style,))
        ret[name] = Colorizer(name=name, color=fg, background=bg, underline=underline)
    return ret


globals().update((name.upper(), Colorizer(color=name, name=name)) for name in COLORS)


def main():
    """
    Colorize lines from stdin
    """
    import fileinput
    try:
        for line in fileinput.input(openhook=functools.partial(open, errors='replace')):
            print(colorize(line), end="", flush=True)
    except BrokenPipeError:
        pass


if __name__ == "__main__":
    main()
