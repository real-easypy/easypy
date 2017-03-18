
import itertools

def cartesian(**kwargs):
    """
    Given keyword arguments with iterable values, yields dictionaries of all
    possible keyword/value permutations.

    Example:

        >>> list(cartesian(a=[1,2], b="xyz"))
        [{'b': 'x', 'a': 1},
         {'b': 'x', 'a': 2},
         {'b': 'y', 'a': 1},
         {'b': 'y', 'a': 2},
         {'b': 'z', 'a': 1},
         {'b': 'z', 'a': 2}]

    """
    param_names = kwargs.keys()
    param_ranges = kwargs.values()

    for param_values in itertools.product(*param_ranges):
        params = dict(zip(param_names, param_values))
        yield params
