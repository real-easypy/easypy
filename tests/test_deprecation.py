import pytest
from easypy.deprecation import deprecated_arguments


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_deprecated_arguments():
    @deprecated_arguments(foo='bar')
    def func(bar):
        return 'bar is %s' % (bar,)

    assert func(1) == func(foo=1) == func(bar=1) == 'bar is 1'

    with pytest.raises(TypeError):
        func(foo=1, bar=2)

    with pytest.raises(TypeError):
        func(1, foo=2)
