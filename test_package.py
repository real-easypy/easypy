import pytest
import sys

packages = list(filter(None, """
_multithreading_init
aliasing
bunch
caching
collections
colors
concurrency
contexts
decorations
deprecation
exceptions
fixtures
gevent
humanize
interaction
lockstep
meta
misc
predicates
properties
random
resilience
signals
sync
tables
threadtree
timing
tokens
typed_struct
units
words
ziplog
""".split()))


@pytest.yield_fixture
def clean_modules():
    yield
    roots = "easypy", "logbook", "logging"
    for n in sorted(sys.modules):
        if any(n.startswith(root) for root in roots):
            sys.modules.pop(n)


@pytest.mark.parametrize("package", packages)
def test_package(package, clean_modules):
    __import__("easypy.%s" % package)
