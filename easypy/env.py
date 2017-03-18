import os
import inspect
from collections import MutableMapping
from easypy.misc import Token
from easypy.humanize import yesno_to_bool


NO_DEFAULT = Token("REQUIRED")


class Env(MutableMapping):
    """
    This object can be used in 'exploratory' mode:

        nv = Env()
        print(nv.HOME)

    It can also be used as a parser and validator of environment variables:

    class MyEnv(Env):
        username = Env.Str("USER", default='anonymous')
        path = Env.CSV("PATH", separator=":")
        tmp = Env.Str("TEMP")  # required

    nv = MyEnv()

    print(nv.username)

    for p in nv.path:
        print(p)

    try:
        print(p.tmp)
    except KeyError:
        print("TEMP is not defined")
    else:
        assert False
    """

    __slots__ = ["_env", "_defined_keys"]

    class _BaseVar(object):

        def __init__(self, name, default=NO_DEFAULT):
            self.name = name
            self.default = default

        def convert(self, value):
            return value

        def __get__(self, instance, owner):
            try:
                return self.convert(instance._raw_get(self.name))
            except KeyError:
                if self.default is NO_DEFAULT:
                    raise
                return self.default

        def __set__(self, instance, value):
            instance[self.name] = value

    class Str(_BaseVar):
        pass

    class Bool(_BaseVar):
        convert = staticmethod(yesno_to_bool)

        def __set__(self, instance, value):
            instance[self.name] = "yes" if value else "no"

    class Int(_BaseVar):
        convert = staticmethod(int)

    class Float(_BaseVar):
        convert = staticmethod(float)

    class CSV(_BaseVar):

        def __init__(self, name, default=NO_DEFAULT, type=str, separator=","):
            super().__init__(name, default=default)
            self.type = type
            self.separator = separator

        def __set__(self, instance, value):
            instance[self.name] = self.separator.join(map(str, value))

        def convert(self, value):
            return [self.type(v.strip()) for v in value.split(self.separator)]

    # =========

    def __init__(self, env=os.environ):
        self._env = env
        self._defined_keys = {k for (k, v) in inspect.getmembers(self.__class__) if isinstance(v, self._BaseVar)}

    def __iter__(self):
        return iter(dir(self))

    def __len__(self):
        return len(self._env)

    def __delitem__(self, name):
        del self._env[name]

    def __setitem__(self, name, value):
        self._env[name] = str(value)

    def _raw_get(self, key):
        return self._env[key]

    def __contains__(self, key):
        try:
            self._raw_get(key)
        except KeyError:
            return False
        else:
            return True

    def __getattr__(self, name):
        # if we're here then there was no descriptor defined
        try:
            return self._raw_get(name)
        except KeyError:
            raise AttributeError("%s has no attribute %r" % (self.__class__, name))

    def __getitem__(self, key):
        return getattr(self, key)  # delegate through the descriptors

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def __dir__(self):
        if self._defined_keys:
            # return only defined
            return sorted(self._defined_keys)
        # return whatever is in the environemnt (for convenience)
        members = set(self._env.keys())
        members.update(object.__dir__(self))
        return sorted(members)


def test_env():

    import pytest

    class E(Env):
        terminal = Env.Str("TERM")
        B = Env.Bool("BOOL", default=True)
        I = Env.Int("INT")
        INTS = Env.CSV("CS_INTS", type=int)

    raw_env = dict(TERM="xterm", CS_INTS="1,2,3,4")
    e = E(raw_env)

    assert e.terminal == "xterm"
    e.terminal = "foo"
    assert e.terminal == "foo"
    assert raw_env["TERM"] == "foo"
    assert "terminal" not in raw_env

    # check default
    assert e.B is True

    raw_env['BOOL'] = "no"
    assert e.B is False

    raw_env['BOOL'] = "0"
    assert e.B is False

    e.B = True
    assert raw_env['BOOL'] == "yes"

    e.B = False
    assert raw_env['BOOL'] == "no"

    assert e.INTS == [1, 2, 3, 4]
    e.INTS = [1, 2]
    assert e.INTS == [1, 2]
    e.INTS = [1, 2, 3, 4]

    with pytest.raises(KeyError):
        e.I

    e.I = "5"

    assert raw_env['INT'] == "5"
    assert e.I == 5
    assert e['I'] == 5

    assert "{I} {B} {terminal}".format(**e) == "5 False foo"
    assert dict(e) == dict(I=5, B=False, terminal='foo', INTS=[1, 2, 3, 4])

    r = Env(raw_env)
    assert "{INT} {BOOL} {TERM}".format(**r) == "5 no foo"
