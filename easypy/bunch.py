from easypy.exceptions import TException

# TODO: some methods like fromkeys/update can insert keys which are not strings - and that causes problems.
#       need to either convert keys to strings (like in _convert), or raise ValueError.
class Bunch(dict):

    __slots__ = ("__stop_recursing__",)

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            if name[0] == "_" and name[1:].isdigit():
                return self[name[1:]]
            raise AttributeError("%s has no attribute %r" % (self.__class__, name))

    def __getitem__(self, key):
        try:
            return super(Bunch, self).__getitem__(key)
        except KeyError:
            from numbers import Integral
            if isinstance(key, Integral):
                return self[str(key)]
            raise

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        try:
            del self[name]
        except KeyError:
            raise AttributeError("%s has no attribute %r" % (self.__class__, name))

    def __getstate__(self):
        return self

    def __setstate__(self, dict):
        self.update(dict)

    def __repr__(self):
        if getattr(self, "__stop_recursing__", False):
            items = sorted("%s" % k for k in self if isinstance(k, str) and not k.startswith("__"))
            attrs = ", ".join(items)
        else:
            dict.__setattr__(self, "__stop_recursing__", True)
            try:
                attrs = self.render()
            finally:
                dict.__delattr__(self, "__stop_recursing__")
        return "%s(%s)" % (self.__class__.__name__, attrs)

    def render(self):
        items = sorted("%s=%r" % (k, v) for k, v in self.items() if isinstance(k, str) and not k.startswith("__"))
        return ", ".join(items)

    def _repr_pretty_(self, *args, **kwargs):
        from easypy.humanize import ipython_mapping_repr
        return ipython_mapping_repr(self, *args, **kwargs)

    def to_dict(self):
        return unbunchify(self)

    def to_json(self):
        import json
        return json.dumps(self.to_dict())

    def to_yaml(self):
        import yaml
        return yaml.dump(self.to_dict())

    def copy(self, deep=False):
        if deep:
            return _convert(self, self.__class__)
        else:
            return self.__class__(self)

    @classmethod
    def from_dict(cls, d):
        return _convert(d, cls)

    @classmethod
    def from_json(cls, d):
        import json
        return cls.from_dict(json.loads(d))

    @classmethod
    def from_yaml(cls, d):
        import yaml
        return cls.from_dict(yaml.load(d))

    @classmethod
    def from_xml(cls, d):
        import xmltodict
        return cls.from_dict(xmltodict.parse(d))

    def __dir__(self):
        members = set(k for k in self if isinstance(k, str) and (k[0] == "_" or k.replace("_", "").isalnum()))
        members.update(dict.__dir__(self))
        return sorted(members)

    def without(self, *keys):
        "Return a shallow copy of the bunch without the specified keys"
        return Bunch((k, v) for k, v in self.items() if k not in keys)

    def but_with(self, **kw):
        "Return a shallow copy of the bunch with the specified keys"
        return Bunch(self, **kw)


def _convert(d, typ):
    if isinstance(d, dict):
        return typ(dict((str(k), _convert(v, typ)) for k, v in d.items()))
    elif isinstance(d, (tuple, list, set)):
        return type(d)(_convert(e, typ) for e in  d)
    else:
        return d


def unbunchify(d):
    """Recursively convert Bunches in `d` to a regular dicts."""
    return _convert(d, dict)


def bunchify(d=None, **kw):
    """Recursively convert dicts in `d` to Bunches.
    If `kw` given, recursively convert dicts in it to Bunches and update `d` with it.
    If `d` is None, an empty Bunch is made."""
    d = _convert(d, Bunch) if d is not None else Bunch()
    if kw:
        d.update(bunchify(kw))
    return d
