# -*- coding: utf-8 -*-
from __future__ import absolute_import
import os
import threading
import time
import itertools
import codecs
import re
import traceback
from math import ceil
from collections import namedtuple
from contextlib import contextmanager
from inspect import getargspec
from io import StringIO
from datetime import datetime, timedelta

from easypy.bunch import Bunch, bunchify
from easypy.colors import Colorizer
from easypy.collections import ilistify
from easypy.misc import clamp


def compact(line, length, ellipsis="....", suffix_length=20):
    if len(line) <= length:
        return line
    return line[:length-len(ellipsis)-suffix_length] + ellipsis + line[-suffix_length:]


is_non_printable = (set(range(32)) | set(range(127, 256))).__contains__

printables = set(map(ord, '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ '))


def is_printable(char, include_space=True):
    """
    Returns ``True`` if the character is printable and NOT a control character.
    """
    if char == " " and not include_space:
        return False
    return char in printables


def bool_to_yesno(b):
    return "yes" if b else "no"


def yesno_to_bool(s):
    s = s.lower()
    if s not in ("yes", "no", "true", "false", "1", "0"):
        raise ValueError("Unrecognized boolean value: %r" % (s,))
    return s in ("yes", "true", "1")


def time_duration(delta, ago=False):
    if not isinstance(delta, timedelta):
        delta = timedelta(seconds=delta)

    negative = False
    if delta.total_seconds() < 0:
        negative = True
        delta = timedelta(seconds=-delta.total_seconds())

    days = delta.days
    hours, seconds = divmod(delta.seconds, 60*60)
    minutes, seconds = divmod(seconds, 60)

    if delta.total_seconds() <= 2:
        if ago:
            txt = "just now"
        else:
            txt = "no-time"
    else:
        if days:
            units = "a", "day", days, "hour", hours, 24
        elif hours:
            units = "an", "hour", hours, "minute", minutes, 60
        else:
            units = "a", "minute", minutes, "second", seconds, 60

        article, major_unit, major, minor_unit, minor, ratio = units
        distance = minor / ratio
        if distance > 0.6:
            major += 1
            minor -= ratio
            distance -= 1

        if not major:
            txt = "{minor} {minor_unit}s"
        elif major == 1:
            if distance <= -0.1:
                txt = "almost {article} {major_unit}"
            elif distance <= 0.1:
                txt = "{article} {major_unit}"
            elif distance <= 0.3:
                txt = "about {article} {major_unit}"
            elif distance <= 0.7:
                txt = "{major}½ {major_unit}s"
        else:
            if distance <= -0.1:
                txt = "almost {major} {major_unit}s"
            elif distance <= 0.1:
                txt = "{major} {major_unit}s"
            elif distance <= 0.3:
                txt = "about {major} {major_unit}s"
            elif distance <= 0.7:
                txt = "{major}½ {major_unit}s"
        txt = txt.format(**locals())
        if ago:
            txt += " from now" if negative else " ago"
    return txt


def from_utc(utc_str):
    return datetime.strptime(utc_str + "+0000", '%Y-%m-%dT%H:%M:%S.%fZ%z').timestamp()


def time_ago(t, now=None):
    now = datetime.fromtimestamp(now) if now else datetime.now()
    delta = now - datetime.fromtimestamp(t)
    return time_duration(delta, ago=True)


INPUT_TIME_PATTERNS = [
    ((re.compile(p[0]),) + tuple(p[1:])) for p in (
        (r"(\d{1,2}/\d{1,2}-\d{1,2}:\d{1,2}:\d{1,2})", "%d/%m-%H:%M:%S", 1, 0),        # 21/05-17:09:59
        (r"(\d{4}-\d{1,2}-\d{1,2}(?P<sep>[T-])\d{1,2}:\d{1,2}:\d{1,2})(?:\d*\.\d*)?(?P<utc>Z)?",
            "%Y-%m-%d{sep}%H:%M:%S", 0, time.timezone),
        (r"(\d{4}-\d{1,2}-\d{1,2}-\d{1,2}:\d{1,2})", "%Y-%m-%d-%H:%M", 0, 0),
        (r"(\d{1,2}-\d{1,2}-\d{1,2}:\d{1,2})", "%m-%d-%H:%M", 1, 0),
        (r"(\w+\s+\d{1,2}\s+\d{1,2}:\d{1,2}:\d{1,2})", "%b %d %H:%M:%S", 1, 0),
        (r"(\d{1,2}-\d{1,2}:\d{1,2}:\d{1,2})", "%d-%H:%M:%S", 2, 0),
        (r"(\d{1,2}-\d{1,2}:\d{1,2})", "%d-%H:%M", 2, 0),
        (r"(\d{1,2}:\d{1,2}:\d{1,2})", "%H:%M:%S", 3, 0),
        (r"(\d{1,2}:\d{1,2})", "%H:%M", 3, 0),
    )]


def parse_fuzzy_time(ts, baseline=None):
    for regex, fmt, missing, offset in INPUT_TIME_PATTERNS:
        match = regex.match(ts)
        if match:
            break
    else:
        raise ValueError("Invalid time pattern: %r (must be one of: [%s])" % (ts, ", ".join(a[1] for a in INPUT_TIME_PATTERNS)))

    ts = match.group(1)
    params = match.groupdict()
    if not params.get('utc'):
        offset = 0
    time_tuple = time.strptime(ts, fmt.format(**params))
    if missing:
        baseline = time.localtime(baseline)
        time_tuple = baseline[:missing] + time_tuple[missing:]
    return time.mktime(time_tuple) - offset


Node = namedtuple("Node", "fmt args kwargs children")


class IndentableTextBuffer():
    NICE_BOX = bunchify(
        LINE=            "─",
        INDENT_SEGMENT=  "│   ",
        OPEN=            "┬",
        INDENT_OPEN=     "├───",
        INDENT_CLOSE=    "╰",
        SEGMENT_END=     "╼ ",
        SEGMENT_START=   " ╾",
        SECTION_OPEN=    "╮",
        SECTION_SEGMENT= "│",
        SECTION_CLOSE=   "╯",
        )
    TEXTUAL_BOX = bunchify(
        LINE=            "-",
        INDENT_SEGMENT=  "|   ",
        INDENT_OPEN=     "+---",
        OPEN=            ".",
        INDENT_CLOSE=    "`",
        SEGMENT_END=     "- ",
        SEGMENT_START=   " -",
        SECTION_OPEN=    ".",
        SECTION_SEGMENT= "|",
        SECTION_CLOSE=   "*",
        )

    def __init__(self, fmt="", *args, **kwargs):
        self.current = self.root = Node(fmt, args, kwargs, [])

    def __repr__(self):
        return self.render(width=80)

    def __len__(self):
        def count(root):
            for child in root.children:
                if isinstance(child, str):
                    yield 1
                else:
                    yield from count(child)
        return sum(count(self.root))

    def write(self, line, *args, **kwargs):
        if args or kwargs:
            line = line.format(*args, **kwargs)
        self.current.children.append(line)

    def extend(self, other):
        self.current.children.extend(other.root.children)

    @contextmanager
    def indent(self, fmt, *args, **kwargs):
        parent = self.current
        self.current = Node(fmt, args, kwargs, [])
        parent.children.append(self.current)
        yield
        self.current = parent

    def render(self, width=None, textual=None, prune=False, file=None, overflow='ignore', edges=True):
        if width is None:
            from .logging import TERM_WIDTH as width
        if textual is None:
            from .logging import GRAPHICAL
            textual = not GRAPHICAL

        buff = file if file else StringIO()
        G = self.TEXTUAL_BOX if textual else self.NICE_BOX
        from textwrap import wrap

        def has_descendents(elem):
            if isinstance(elem, str):
                return True
            return any(map(has_descendents, elem.children))

        def write_tree(elem, depth=0):
            if isinstance(elem, str):
                prefix = G.INDENT_SEGMENT * (depth-1)
                prefix += G.INDENT_SEGMENT
                for par in elem.splitlines():
                    if overflow == "wrap":
                        lines = wrap(par, width-len(prefix)-1)
                    elif overflow == "trim":
                        lines = [compact(par, width-len(prefix)-1)]
                    else:
                        lines = [par]
                    for line in lines:
                        line = prefix + line
                        if len(line) < width:
                            if edges:
                                line = line.ljust(width-1) + G.SECTION_SEGMENT
                        buff.write(line + "\n")
            elif not prune or has_descendents(elem):
                if depth:
                    prefix = G.INDENT_SEGMENT * (depth-1) + G.INDENT_OPEN
                else:
                    prefix = ""
                txt = (G.SEGMENT_END + elem.fmt.format(*elem.args, **elem.kwargs) + G.SEGMENT_START) if elem.fmt else ""
                header = prefix + G.OPEN + txt
                buff.write(header.ljust(width-1, G.LINE) + G.SECTION_OPEN + "\n")
                for child in elem.children:
                    write_tree(child, depth+1)
                footer = (G.INDENT_SEGMENT*depth + G.INDENT_CLOSE + G.LINE * (width-len(G.INDENT_SEGMENT)*(depth+1)-len(txt)+2) + txt)
                buff.write(footer + G.SECTION_CLOSE + "\n")

        write_tree(self.root)
        if not file:
            return buff.getvalue()


def format_in_columns(elements, total_width=None, sep="  ", indent="  ", min_height=10):
    """
    >>> print(format_in_columns([str(i) for i in range(100)], 50))
      0  10  20  30  40  50  60  70  80  90
      1  11  21  31  41  51  61  71  81  91
      2  12  22  32  42  52  62  72  82  92
      3  13  23  33  43  53  63  73  83  93
      4  14  24  34  44  54  64  74  84  94
      5  15  25  35  45  55  65  75  85  95
      6  16  26  36  46  56  66  76  86  96
      7  17  27  37  47  57  67  77  87  97
      8  18  28  38  48  58  68  78  88  98
      9  19  29  39  49  59  69  79  89  99
    """

    if not total_width:
        try:
            total_width, _ = os.get_terminal_size()
        except:
            total_width = 80

    widest = min(max(len(k) for k in elements), total_width)
    columns = max((total_width - len(indent)) // (widest + len(sep)), 1)
    height = max(min_height, (len(elements) // columns) + 1)

    # arrange the elements in columns
    columns = [[elem for (__, elem) in group]
               for __, group in itertools.groupby(enumerate(elements), lambda p: p[0]//height)]
    rows = itertools.zip_longest(*columns)

    col_max = total_width - len(sep) * (len(columns) - 1)
    column_lens = [min(max(map(len, column)), col_max) for column in columns]

    return '\n'.join(indent + sep.join([(string or "").ljust(column_lens[column_num])
                                        for column_num, string in enumerate(row)])
                     for row in rows)


def format_dict(obj, max_width=200, indent="  "):
    key_width = val_width = 0
    items = []
    key_fmt, sep, format_output = ((str, " = ", "\n{0.__class__.__name__}(\n{1}\n)".format)
                                   if isinstance(obj, Bunch) else
                                   (repr, " : ", "\n{{\n{1}\n}}".format))
    for key, val in obj.items():
        key = key_fmt(key)
        val = repr(val)
        items.append((key, val))
        # figure out widest key/val
        key_width = max(key_width, len(key))
        val_width = max(val_width, len(val), 5)

    # truncate to terminal width
    key_width = min(key_width, max_width - len(sep))
    val_width = min(val_width, max_width - len(sep) - key_width - len(indent))
    elements = ["{:{key_width}}{}{:{val_width}},".format(k, sep, v, key_width=key_width, val_width=val_width)
                for k, v in sorted(items)]
    return format_output(obj, format_in_columns(elements, max_width, indent=indent))


# HexDump-Related ######################################################################

to_hex = {i: ('%02x' % i)
          for i in range(256)
          }.__getitem__

to_printable = {ch: chr(ch) if is_printable(ch) else '.'
                for ch in range(256)
                }.__getitem__


def format_hex(buff, chunk_size=8):
    return '  '.join(
        ' '.join(to_hex(char)
                 for char in buff[offset : (offset + chunk_size) if chunk_size else None])
        for offset in range(0, len(buff), chunk_size or len(buff))
    )


def format_printable(buff, chunk_size=8):
    return ' '.join(
        ''.join(to_printable(char)
                for char in buff[offset : (offset + chunk_size) if chunk_size else None])
        for offset in range(0, len(buff), chunk_size or len(buff))
    )


def iter_hexdump(data, bytes_per_line=32, chunk_size=4, skip_repeats=True, start_offset=0):
    rows = []
    skipping = False
    previous_content = None
    rows.append(("0x....+",
                 format_hex(range(bytes_per_line), chunk_size=chunk_size).upper().replace(" 0", "  "),
                 format_printable([ord("-")]*bytes_per_line, chunk_size=chunk_size)))
    for offset in range(0, len(data), bytes_per_line):
        content = data[offset:offset+bytes_per_line]
        if skip_repeats and (0 < offset < (len(data)-bytes_per_line)) and content == previous_content:
            if skipping:
                continue
            skipping = True
            row = '*'.ljust(7), "", ""
        else:
            previous_content = content
            skipping = False
            row = ("0x{:04x}:".format(start_offset+offset),
                   format_hex(content, chunk_size),
                   format_printable(content, chunk_size)
                   )
        rows.append(row)
    hex_width = len(rows[0][1]) if rows else 0
    fmt = "{} {:{hex_width}} | {}".format
    return (fmt(*row, hex_width=hex_width) for row in rows)


def hexdump(data, bytes_per_line=32, chunk_size=4, skip_repeats=True, start_offset=0):
    return ('\n'.join(iter_hexdump(data, bytes_per_line, chunk_size, skip_repeats, start_offset)))


def from_hexdump(data):
    from io import StringIO, BytesIO
    buff = BytesIO()
    last_offset = None
    repeats = False
    to_bytes = bytes
    for line in StringIO(data):
        if not line:
            continue
        if line.startswith("0x....+"):
            continue
        parts = line.split()
        offset = parts[0]
        if offset.startswith("*"):
            repeats = True
        else:
            offset = eval(offset[:-1])
            if repeats:
                buff.write(chunks * ((offset-last_offset)//len(chunks)-1))
                repeats = False
            chunks = to_bytes(int(c, 16) for c in itertools.takewhile(lambda c: c != "|", parts[1:]))
            buff.write(chunks)
            last_offset = offset
    return buff.getvalue()


# Stateless encoder/decoder
class HexDumpCodec(codecs.Codec):
    def __init__(self, *args):
        self.args = args

    def decode(self, data, errors='strict'):
        return hexdump(data, *self.args), len(data)

    def encode(self, data, errors='strict'):
        return from_hexdump(data), len(data)


# Register the codec search function
def find_hexdump(encoding):
    """Return the codec for ``HexDump``.
    """
    if encoding.startswith('hexdump'):
        args = list(map(int, encoding.split("_")[1:]))
        return codecs.CodecInfo(
            name='hexdump',
            encode=HexDumpCodec(*args).encode,
            decode=HexDumpCodec(*args).decode,
            )
    return None

codecs.register(find_hexdump)


def to_new_style_formatter(string):
    def repl(matched):
        keyword = matched.group(1)
        if keyword:
            return "{%s}" % keyword.strip('()')
        else:
            return "{%d}" % next(index)
    index = itertools.count()
    return re.sub(r'(?:%(\(\w+\))?[diouxXeEfFgGcrs])', repl, string)


class TrimmingTemplate(str):
    """
    Use new-style string formatting syntax with the old modulu string formatting operator:

    >>> TrimmingTemplate("{b}: {a}") % dict(a=1, b=2)
    '2: 1'


    Use the ``~`` next to the field width specifier so that the content is
    trimmed to fit the specified field width.  Works with string data only.

    >>> TrimmingTemplate("{id:5}:{header:10~} {footer:~11}") % dict(id=225, header='This is not the end', footer='This is not the end!')
    '  225:This is... ...the end!'

    """

    RE_NEW_STYLE_FORMATTERS = re.compile(r"{(\w+):([^}]*)}")

    def __init__(self, s):
        assert s == to_new_style_formatter(s), "Can't use old-style string formatting syntax"
        self.trimmers = {}
        for name, format in self.RE_NEW_STYLE_FORMATTERS.findall(s):
            if format.startswith("~"):
                limiter = lambda s, limit=int(format[1:]): ("..." + s[-(limit-3):]) if len(s) > limit else s
            elif format.endswith("~"):
                limiter = lambda s, limit=int(format[:-1]): s[:limit-3] + "..." if len(s) > limit else s
            else:
                continue
            self.trimmers[name] = limiter
        self._fmt = self.replace("~", "")

    def __mod__(self, other):
        if isinstance(other, dict):
            return self.format(**other)
        elif isinstance(other, tuple):
            return self.format(*other)
        else:
            return self.format(other)

    def format(self, **kwargs):
        kwargs = {k: self.trimmers.get(k, lambda s: s)(v) for (k, v) in kwargs.items()}
        return self._fmt.format(**kwargs)


# Tracebacks
def get_traceback_formatter(default_root_path=None):
    from plumbum import local
    if default_root_path:
        def relative(path):
            r = local.path(path) - default_root_path
            if r and r[0] != "..":
                path = "./%s" % r
            return path
        default_root_path = local.path(default_root_path)
    else:
        relative = lambda path: path

    def _format_list_iter(extracted_list):
        extracted_list = list(extracted_list)
        lines = []
        for filename, lineno, name, line in extracted_list:
            filename = relative(filename)
            left = "  {}:{} ".format(filename, lineno)
            right = " {}".format(name)
            lines.append((len(left)+len(right), left, right, line))

        width = max(args[0] for args in lines) + 4
        for _, left, right, line in lines:
            item = left.ljust(width-len(right), ".") + right
            if line:
                item = item + ' >> {}'.format(line.strip())
            yield item + '\n'
    return _format_list_iter


def format_tb(*args, **kw):
    import traceback
    default_root_path = kw.pop("default_root_path", None)
    orig, traceback._format_list_iter = traceback._format_list_iter, get_traceback_formatter(default_root_path=default_root_path)
    try:
        return traceback.format_tb(*args)
    finally:
        traceback._format_list_iter = orig


def format_thread_stack(frame, skip_modules=[threading]):
    stack = traceback.extract_stack(frame)
    if skip_modules:
        itr_stack = iter(stack)
        items = []
        fnames = {m.__file__ for m in ilistify(skip_modules)}
        # skip everything until after specified module
        for fname, *_ in itr_stack:
            if fname in fnames:
                items.append([fname] + _)
                for i, (fname, *_) in enumerate(itr_stack, 1):
                    if fname not in fnames:
                        if i > 1:
                            items.append([last_fname, "...(%s)" % i, "---", "---"])
                        break
                    last_fname = fname
            items.append([fname] + _)
        if len(items) <= 2:
            items = stack
    else:
        items = stack

    return ''.join(traceback.format_list(items))


def format_table(table, titles=True):
    """
    Returns a multilined string representing the given table (2d list) as a table, with equal-width columns.

    :param titles: if true, the first row in the table is taken as headers for
    the table, adding a separator on the second line
    """

    fmt_simple = "{:{width}}"
    fmt_string = "{!s:{width}}"  # for type that don't accept a ':width' specifier when formatted (NoneType, dict, ...)

    def safe_format(cell, width=1):
        if isinstance(cell, bool):
            # because if we force it to str it becomes an int (True -> 1, False -> 0)
            return fmt_string.format(cell, width=width)

        try:
            return fmt_simple.format(cell, width=width)
        except TypeError:
            return fmt_string.format(cell, width=width)

    widths = [max(map(len, map(safe_format, column))) for column in zip(*table)]
    txt = ''
    for i, row in enumerate(table):
        if titles and i == 1:
            txt += '-'.join('-' * width for width in widths) + '\n'
        txt += '|'.join(safe_format(cell, width=width) for cell, width in zip(row, widths))
        txt += '\n'
    return txt


def format_size(num, suffix='B'):
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)


def ipython_fields_repr(name, field_items, p, cycle):
    from easypy.colors import DARK_CYAN
    if cycle:
        p.text('%s(...)' % (name,))
        return
    prefix = '%s(' % (name,)
    with p.group(len(prefix), prefix, ')'):
        for idx, (k, v) in enumerate(field_items):
            if idx:
                p.text(',')
                p.breakable()
            with p.group(len(k)+1, DARK_CYAN(k) + "="):
                p.pretty(v)


def ipython_mapping_repr(mapping, p, cycle):
    """Used by IPython. add to any mapping class as '_repr_pretty_'"""
    ipython_fields_repr(mapping.__class__.__name__, sorted(mapping.items()), p, cycle)


BAR_SEQUENCE = ' ▏▎▍▌▋▊▉██'
VERT_SEQUENCE = ' ▁▂▃▄▅▆▇█'
LITTLE_DIGITS = '⁰¹²³⁴⁵⁶⁷⁸⁹'
RULER_MARKS = '0' + LITTLE_DIGITS[1:]


def vertbar(ratio):
    ratio = max(min(1.0, ratio), 0.0)
    return VERT_SEQUENCE[ceil(clamp(ratio, 0, 1) * (len(VERT_SEQUENCE)-1))]


def horizbar(ratio, width=2):
    ratio = max(min(1.0, ratio), 0.0) * width
    return "".join(
        BAR_SEQUENCE[
            ceil(clamp(ratio-p, 0, 1) * (len(BAR_SEQUENCE)-1))
            ]
        for p in range(width))


def name_generator():
    from string import digits
    from itertools import product

    easynames = ('alfa bravo charlie delta echo foxtrot golf hotel india juliett kilo lima mike november '
                 'oscar papa quebec romeo sierra tango uniform victor whiskey xray yankee zulu').split()

    for parts in product(easynames, easynames, digits):
        yield "_".join(parts)


def percentages_comparison(actual, expected, key_caption='Item', color_bounds={'green': 0,
                                                                               'yellow': 5,
                                                                               'red': 10}):
    from easypy.tables import Table, Column
    assert all(0 <= count for count in actual.values())
    assert all(0 <= count for count in expected.values())

    table = Table()
    table.add_column(Column("key", title=key_caption))
    table.add_column(Column("actual", title='Actual'))
    table.add_column(Column("expected", title='Expected'))
    table.add_column(Column("diff", title='Difference'))

    actual_total = sum(actual.values())
    expected_total = sum(expected.values())
    if actual_total == 0 or expected_total == 0:
        return table

    color_bounds = sorted(color_bounds.items(), key=lambda a: a[1], reverse=True)

    def color_for(percentage):
        for color, bound in color_bounds:
            if bound <= percentage:
                return color

    def generate():
        for key in actual.keys() | expected.keys():
            actual_count = actual.get(key, 0)
            expected_count = expected.get(key, 0)
            if actual_count == 0 and expected_count == 0:
                continue
            actual_percentage = 100 * actual_count / actual_total
            expected_percentage = 100 * expected_count / expected_total
            diff = actual_percentage - expected_percentage
            abs_diff = abs(diff)
            color = color_for(abs_diff)
            if color:
                colorizer = Colorizer(color)
                yield Bunch(key=colorizer(key),
                            actual=colorizer('%.1f%%' % actual_percentage),
                            expected=colorizer('%.1f%%' % expected_percentage),
                            diff=colorizer('%.1f%%' % diff),
                            abs_diff=abs_diff)

    for item in sorted(generate(), key=lambda a: a.abs_diff, reverse=True):
        table.add_row(**item)
    return table


class _ReprAsString:
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return str(self.value)


def easy_repr(*attributes):
    """
    Create a simple __repr__ function for the decorated class.
    """

    def _decorator(cls):
        assert attributes, 'must provide at least one attribute'

        def _nice_repr(self):
            attrs = ', '.join('{}={!r}'.format(attr, getattr(self, attr)) for attr in attributes)
            return '<{self.__class__.__name__}: {attrs}>'.format(self=self, attrs=attrs)
        cls.__repr__ = _nice_repr
        return cls
    return _decorator


class Histogram():
    """
    Simple management of saving and displaying histogram values
    """

    def __init__(self, bins):
        """
        :param bins: a list of bin values
        """
        self._bins = sorted(bins)
        self._values = [0] * (len(bins) + 1)

    def __str__(self):
        return '%s, %s' % (self._values, self._bins)

    def increment(self, value):
        # could be improved by using binary search assuming `bins` is already sorted
        for i, current_bin in enumerate(self._bins):
            if value < current_bin:
                self._values[i] += 1
                return

        self._values[-1] += 1

    def get_values(self):
        return self._values[:]

    def show_graph(self, writeline_callback, size):
        """
        Shows a nice graph of the current values

        :param writeline_callback: a callback which can take a format string and arguments (fmt, *args)
        :param size: the maximum size of each bar
        """

        def make_bar(val, ratio):
            bar_length = int(val * ratio)
            if bar_length == 0 and val > 0:
                bar_length = 1

            return '-' * bar_length

        if len(self._bins) < 1:
            return

        # check both min and max in case of negative numbers
        max_bin_length = max(len(str(max(self._bins))), len(str(min(self._bins))))
        first_record_format = '{}<%-{}d| %5.2f%% | %s'.format(' ' * max_bin_length, max_bin_length)
        mid_record_format = '%{}d-%-{}d| %5.2f%% | %s'.format(max_bin_length, max_bin_length)
        last_record_format = '%{}d<{}| %5.2f%% | %s'.format(max_bin_length, ' ' * max_bin_length)
        max_val = max(self._values)
        arr_sum = sum(self._values)
        if max_val > 0:
            ratio = size / max_val
        else:
            ratio = 0

        writeline_callback(
            first_record_format,
            self._bins[0], self._values[0] / arr_sum * 100, make_bar(self._values[0], ratio))

        if len(self._bins) > 1:
            for i in range(1, len(self._bins)):
                writeline_callback(
                    mid_record_format,
                    self._bins[i - 1], self._bins[i], self._values[i] / arr_sum * 100, make_bar(self._values[i], ratio))

        writeline_callback(
            last_record_format,
            self._bins[-1], self._values[-1] / arr_sum * 100, make_bar(self._values[-1], ratio))
