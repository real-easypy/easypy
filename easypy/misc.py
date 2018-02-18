import inspect


def strip_html(s):
    from xml.etree.ElementTree import fromstring
    from html import unescape
    return unescape("".join(fromstring("<root>%s</root>" % s).itertext()))


class Hex(int):

    def __str__(self):
        return "%X" % self

    def __repr__(self):
        return "0x%x" % self


class Token(str):

    _all = {}

    def __new__(cls, name):
        name = name.strip("<>")
        try:
            return cls._all[name]
        except KeyError:
            pass
        cls._all[name] = self = super().__new__(cls, "<%s>" % name)
        return self

    def __repr__(self):
        return self

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self is other
        elif isinstance(other, str):
            return self.strip("<>").lower() == other.strip("<>").lower()
        return False

    # we're already case insensitive when comparing
    def lower(self): return self

    def upper(self): return self

    def __hash__(self):
        return super().__hash__()


def __LOCATION__():
    frame = inspect.getframeinfo(inspect.stack()[1][0])
    return "%s @ %s:%s" % (frame.function, frame.filename, frame.lineno)


def get_all_subclasses(cls, include_mixins=False):
    _all = cls.__subclasses__() + [rec_subclass
                                   for subclass in cls.__subclasses__()
                                   for rec_subclass in get_all_subclasses(subclass, include_mixins=include_mixins)]
    if not include_mixins:
        return [subclass for subclass in _all if not hasattr(subclass, "_%s__is_mixin" % subclass.__name__)]
    else:
        return _all


def stack_level_to_get_out_of_file():
    frame = inspect.currentframe().f_back
    filename = frame.f_code.co_filename
    stack_levels = 1
    while frame.f_code.co_filename == filename:
        stack_levels += 1
        frame = frame.f_back
    return stack_levels


def clamp(val, *, at_least=None, at_most=None):
    "Clamp a value so it doesn't exceed specified limits"

    if at_most is not None:
        val = min(at_most, val)
    if at_least is not None:
        val = max(at_least, val)
    return val
