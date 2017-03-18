def _build_param_string(args, kwargs):
    return ", ".join([repr(a) for a in args] + ["%s=%r" % (k, v) for k, v in kwargs.items()])


def print_assert_called_with(mockObject, name):
    for args, kwargs in getattr(mockObject, 'call_args_list', ()):
        print("%s.assert_called_with(%s)" % (name, _build_param_string(args, kwargs)))


def print_assert_any_call(mockObject, name):
    for args, kwargs in getattr(mockObject, 'call_args_list', ()):
        print("%s.assert_any_call(%s)" % (name, _build_param_string(args, kwargs)))


def print_assert_called_with_methods(mockObject, name):
    for method, args, kwargs in getattr(mockObject, 'mock_calls', ()):
        method = method.replace("()", ".return_value")
        call = ".".join(n for n in [name, method, "assert_called_with"] if n)
        print("%s(%s)" % (call, _build_param_string(args, kwargs)))


class MultipleReturnValues(object):
    """
    Allows returning diferent return values on different calls to a mock.

    >>> import mock
    >>> obj = mock.Mock()
    >>> obj.foo.side_effect = MultipleReturnValues(5, 7, 9)
    >>> obj.foo()
    5
    >>> obj.foo()
    7
    >>> obj.foo()
    9
    """
    def __init__(self, *values):
        self.values = list(values)
        self.iter = iter(self.values)

    def __call__(self, *args):
        return self.iter.next()

    def append(self, value):
        self.values.append(value)
