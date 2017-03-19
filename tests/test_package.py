import easypy
import pkgutil
import pytest

packages = [name for ff, name, is_pkg in pkgutil.walk_packages(easypy.__path__)]


@pytest.mark.parametrize("package", packages)
def test_package(package):
    __import__("easypy.%s" % package)
