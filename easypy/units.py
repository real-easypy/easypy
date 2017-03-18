import re
from itertools import repeat
import random
import numbers
from easypy.exceptions import TException


class UnknownDataSizeError(TException):
    template = 'Could not parse size {}'


class DataSize(int):
    BINARY_UNITS = {unit_name: 1024 ** i for i, unit_name in enumerate(['byte', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB', 'XiB'])}
    DECIMAL_UNITS = {unit_name: 1000 ** i for i, unit_name in enumerate(['KB', 'MB', 'GB', 'TB', 'PB', 'XB'], 1)}
    NAMED_UNITS = dict(BINARY_UNITS, **DECIMAL_UNITS)
    UNIT_NAMES = {unit: name for name, unit in NAMED_UNITS.items()}
    SORTED_UNITS = sorted(BINARY_UNITS.values(), reverse=True)

    def __new__(cls, value):
        if (isinstance(value, str)):
            i, u = re.match("(\d*(?:\.\d+)?)?(\w*)", value).groups()
            i = 1 if not i else float(i) if "." in i else int(i)
            value = i * (cls.NAMED_UNITS[u] if u else 1)
            if (isinstance(value, cls)):
                return value

        return super(DataSize, cls).__new__(cls, value)

    def __abs__(self):
        return self.__class__(int.__abs__(self))

    def __add__(self, other):
        return self.__class__(int.__add__(self, other))
    __radd__ = __add__

    def __sub__(self, other):
        return self.__class__(int.__sub__(self, other))

    def __rsub__(self, other):
        return (-self) + other

    def __mul__(self, mul):
        if isinstance(mul, self.__class__):
            raise TypeError('Cannot multiply %s by %s' % (self, mul))
        if not isinstance(mul, numbers.Real):
            raise TypeError('Cannot multiply %s by %r' % (self, mul))
        return self.__class__(int(self) * mul)
    __rmul__ = __mul__

    def __floordiv__(self, div):
        res = int.__floordiv__(self, div)
        if not isinstance(div, self.__class__):
            # if div is not a datasize, return a datasize
            return self.__class__(res)
        # otherwise return an int
        return res

    def __rfloordiv__(self, div):
        res = int.__rfloordiv__(self, div)
        if not isinstance(div, self.__class__):
            # if div is not a datasize, return a datasize
            return self.__class__(res)
        # otherwise return an int
        return res

    def __mod__(self, div):
        return self.__class__(int.__mod__(self, div))

    def __rmod__(self, number):
        if number != 0:
            raise ArithmeticError("Cannot perform modulu of %s by %s" % (number, self))
        return 0

    def __neg__(self):
        return -1*self

    def rounddown(self, unit):
        return (self // unit) * unit

    def roundup(self, unit):
        if 0 == self % unit:
            return self
        else:
            return (self // unit + 1) * unit

    def round(self, unit):
        if self % unit > unit / 2:
            add = unit
        else:
            add = 0
        return self.rounddown(unit) + add

    def ceildiv(self, other):
        return self.roundup(other) / other

    def randrange(self, divisor, start=None):
        'returns a random size below self that divides by divisor'
        if start is None:
            start = self.__class__(0)
        start = start.ceildiv(divisor)
        stop = self.ceildiv(divisor)
        return random.randrange(start=start, stop=stop) * divisor

    def __format__(self, spec):
        width, precision, name = re.match("(?:(\d+))?(?:\.(\d*))?(\w+)?", spec).groups()
        if name == "d":
            ret = "{:d}".format(int(self))
        elif name == "f":
            ret = "{:.{precision}f}".format(float(self), precision=precision or 1)
        elif name in ("b", "byte", "bytes"):
            ret = "{:d}*bytes".format(int(self))
        elif name:
            ret = "{:.{precision}f}*{unit}".format(self / self.NAMED_UNITS[name], unit=name, precision=precision or 1)
        else:
            ret = repr(self)
        return "{:>{width}}".format(ret, width=width or "")

    def __str__(self):
        if 0 == self:
            return '0'
        if self in self.UNIT_NAMES:
            return '1 %s' % (self.UNIT_NAMES[self],)
        for unit in self.SORTED_UNITS:
            name = self.UNIT_NAMES[unit]
            many = 'bytes' if name == 'byte' else name
            if self % unit == 0:
                return '%d%s' % (self/unit, many)
            if self >= unit or (self >= unit/10 and self*10 % unit == 0):
                return '~%.1f%s' % (self/unit, many)
        assert False, "This Should not happen"

    def __repr__(self):
        if self == 0:
            return '0'
        if self in self.UNIT_NAMES:
            return self.UNIT_NAMES[self]
        for unit in self.SORTED_UNITS:
            name = self.UNIT_NAMES[unit]
            many = 'bytes' if name == 'byte' else name
            if self % unit == 0:
                return '%d*%s' % (self / unit, many)
        assert False, "This Should not happen"

    def _repr_pretty_(self, p, cycle):
        # used by IPython
        from easypy.colors import BLUE
        if cycle:
            p.text('DataSize(...)')
            return
        p.text(BLUE(self))


class Duration(float):
    # NAMED_UNITS = reduce(lambda a, b: dict(a, _last=b[0], **{b[0]: a[a['_last']]*int(b[1])}),
    #     (p.split(":") for p in "s:1000 m:60 h:60 d:24".split()),
    #     dict(ms=1, _last='ms'))
    NAMED_UNITS = dict(ms=1/1000, s=1, m=60, h=60*60, d=24*60*60)
    UNIT_NAMES = {unit: name for name, unit in NAMED_UNITS.items()}
    SORTED_UNITS = sorted(NAMED_UNITS.values(), reverse=True)

    def __new__(cls, value):
        if (isinstance(value, str)):
            try:
                i, u = float(value), None
            except ValueError:
                i, u = re.match("(\d*(?:\.\d+)?)?(\w*)", value).groups()
                i = 1.0 if not i else float(i)
            value = i * (cls.NAMED_UNITS[u] if u else 1)
            if (isinstance(value, cls)):
                return value

        return super(Duration, cls).__new__(cls, value)

    def __abs__(self):
        return self.__class__(float.__abs__(self))

    def __add__(self, other):
        return self.__class__(float.__add__(self, other))
    __radd__ = __add__

    def __sub__(self, other):
        return self.__class__(float.__sub__(self, other))

    def __rsub__(self, other):
        return (-self) + other

    def __mul__(self, mul):
        if isinstance(mul, self.__class__):
            raise TypeError('Cannot multiply %s by %s' % (self, mul))
        if not isinstance(mul, numbers.Real):
            raise TypeError('Cannot multiply %s by %r' % (self, mul))
        return self.__class__(super(Duration, self).__mul__(mul))
    __rmul__ = __mul__

    def __truediv__(self, div):
        if isinstance(div, self.__class__):
            return super(Duration, self).__truediv__(div)
        return self * (1/div)

    def __floordiv__(self, div):
        res = float.__floordiv__(self, div)
        if not isinstance(div, self.__class__):
            # if div is not a duration, return a duration
            return self.__class__(res)
        # otherwise return a float
        return res

    def __mod__(self, div):
        return self.__class__(float.__mod__(self, div))

    def __rmod__(self, number):
        if number != 0:
            raise ArithmeticError("Cannot perform modulu of %s by %s" % (number, self))
        return 0

    def __neg__(self):
        return -1*self

    def rounddown(self, unit):
        return (self // unit) * unit

    def roundup(self, unit):
        if 0 == self % unit:
            return self
        else:
            return (self // unit + 1) * unit

    def round(self, unit):
        if self % unit > unit / 2:
            add = unit
        else:
            add = 0
        return self.rounddown(unit) + add

    def ceildiv(self, other):
        return self.roundup(other) / other

    def randrange(self, divisor, start=None):
        'returns a random size below self that divides by divisor'
        if start is None:
            start = self.__class__(0)
        start = start.ceildiv(divisor)
        stop = self.ceildiv(divisor)
        return random.randrange(start=start, stop=stop) * divisor

    def __add__(self, other):
        if isinstance(other, self.__class__):
            return self.__class__(float.__add__(self, other))
        elif isinstance(other, str):
            return self + self.__class__(other)
        else:
            return self.__class__(super(Duration, self).__add__(other))
    __radd__ = __add__

    def __format__(self, spec):
        width, precision, name = re.match("(?:(\d+))?(?:\.(\d*))?(\w+)?", spec).groups()
        if name == "f":
            return super(Duration, self).__format__(spec)
        dir = "<"
        if name == "text":
            from easypy.humanize import time_duration
            ret = time_duration(self)
        elif name == "ago":
            from easypy.humanize import time_duration
            ret = time_duration(self, ago=True)
        else:
            ret = self.render(unit=name, precision=precision)
            dir = ">"
        return "{:{dir}{width}}".format(ret, width=width or "", dir=dir)

    # don't override, so it can be converted to json properly
    # def __str__(self):

    def render(self, unit=None, precision=None):
        if not unit:
            for unit_size in self.SORTED_UNITS:
                if unit_size <= abs(self):
                    break
            unit = self.UNIT_NAMES[unit_size]
        assert unit in self.NAMED_UNITS
        dd = hh = mm = 0
        ss = float(self)
        if unit not in ("ms", "s"):
            mm, ss = divmod(ss, 60)
        if unit != "m":
            hh, mm = divmod(mm, 60)
        if unit != "h":
            dd, hh = divmod(hh, 24)

        if dd or unit == "d":
            return "%.0fd, %02.0f:%02.0fh" % (dd, hh, mm)
        elif hh or unit == "h":
            return "%02.0f:%02.0fh" % (hh, mm)
        elif mm or unit == "m":
            return "%02.0f:%02.0fm" % (mm, ss)
        elif unit in (None, "", "s"):
            return "{:.{precision}f}s".format(ss, precision=precision if precision is not None else 0 if ss.is_integer() else 1)
        else:
            ms = ss * 1000
            return "{:.{precision}f}ms".format(ms, precision=(precision or 1) if ms < 1 else (precision or 0))
        assert False, "This Should not happen"

    __repr__ = render

    def _repr_pretty_(self, p, cycle):
        # used by IPython
        from easypy.colors import MAGENTA
        if cycle:
            p.text('Duration(...)')
            return
        p.text(MAGENTA(self))


byte, KiB, MiB, GiB, TiB, PiB, XiB = repeat(0, 7)  # fixing ugly import error messages and allowing autoimport
KB, MB, GB, TB, PB, XB = repeat(0, 6)

globals().update({name: DataSize(name) for name in DataSize.NAMED_UNITS.keys()})

NANOSECOND = Duration(1/1000000000)
MICROSECOND = Duration(1/1000000)
MILLISECOND = Duration(1/1000)
SECOND = Duration(1)
MINUTE = Duration(60 * SECOND)
HOUR = Duration(60 * MINUTE)
DAY = Duration(24 * HOUR)
WEEK = Duration(7 * DAY)
MONTH = Duration(31 * DAY)
YEAR = Duration(365 * DAY)


# ------


def range_compare(value, value_range):
    if isinstance(value_range, tuple):
        lower, upper = value_range
    else:
        upper = lower = value_range

    if upper is not None:
        if value > upper:
            return value-upper
    if lower is not None:
        if value < lower:
            return value-lower
    return 0


def to_data_size(size):
    from math import ceil
    sizes_table = dict(k=KiB, m=MiB, g=GiB, t=TiB, p=PiB)
    if isinstance(size, (int, float)):
        return size
    try:
        if size[-1].isalpha():
            return ceil(float(size[:-1])) * sizes_table[size[-1].lower()]
        else:
            return ceil(float(size))
    except (KeyError, ValueError):
            raise UnknownDataSizeError(size)
