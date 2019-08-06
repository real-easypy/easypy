import easypy
import pkgutil
import pytest
import sys

packages = [name for ff, name, is_pkg in pkgutil.walk_packages(easypy.__path__)]


@pytest.yield_fixture
def clean_modules():
    yield
    for n in sorted(sys.modules):
        if n.startswith("easypy"):
            sys.modules.pop(n)


@pytest.mark.parametrize("package", packages)
def test_package(package, clean_modules):
    __import__("easypy.%s" % package)
