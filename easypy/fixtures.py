from contextlib import contextmanager
import inspect
from functools import wraps

from easypy.properties import cached_property


class Fixture(object):
    def __init__(self, name, function, cached=True):
        self.name = name
        self._function = function
        self._cached = cached

    def __repr__(self):
        return 'Fixture %s' % self.name

    @cached_property
    def _signature(self):
        return inspect.signature(self._function)

    @cached_property
    def dependencies(self):
        return self._signature.parameters.keys()

    def invoke(self, fixtures_assembly):
        kwargs = {dependency: fixtures_assembly.resolve_fixture(dependency) for dependency in self.dependencies}
        return self._function(**kwargs)


class FixturesNamespace(object):
    def __init__(self):
        self.fixtures = {}

    def register(self, func=None, *, cached=True):
        @wraps(func)
        def inner(func):
            assert func.__name__ not in self.fixtures, 'Fixture %s already exists' % func.__name__
            fixture = Fixture(func.__name__, func, cached=cached)
            self.fixtures[fixture.name] = fixture

        if func is None:
            return inner
        else:
            return inner(func)

    def assemble(self, **manual_fixture_values):
        assembly = FixturesAssembly(self)
        for k, v in manual_fixture_values.items():
            assembly._set_fixture_value(k, v)
        return assembly

    def get(self, name):
        return self.fixtures[name.split('__')[0]]


class FixturesAssembly(object):
    def __init__(self, fixtures_namespace):
        self._fixtures_namespace = fixtures_namespace
        self._fixture_values = {}
        self.__being_added = set()

    @contextmanager
    def __adding(self, name):
        assert name not in self.__being_added, \
            'Circular dependency detected while invoking fixture %s. Currently resolving %s' % (name, self.__being_added)
        self.__being_added.add(name)
        try:
            yield
        finally:
            self.__being_added.remove(name)

    def _set_fixture_value(self, name, value):
        assert name not in self._fixture_values, 'Value for fixture %s already set' % (name,)
        self._fixture_values[name] = value

    def resolve_fixture(self, name):
        try:
            return self._fixture_values[name]
        except KeyError:
            pass

        with self.__adding(name):
            fixture = self._fixtures_namespace.get(name)
            fixture_value = fixture.invoke(self)

        if fixture._cached:
            self._set_fixture_value(name, fixture_value)

        return fixture_value

