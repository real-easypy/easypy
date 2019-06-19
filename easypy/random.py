import string
import random
from random import choice, sample

choose = choice


def random_nice_name(max_length=64, entropy=2, sep='-'):
    """Generates a nice random name from the dictionaries in words

    :param max_length: max length for the name.
    :type max_length: int, optional
    :param entropy: how unique th name will be, currently entropy - 1 adjectives are joined with one noun.
    :type entropy: int, optional
    :param sep: seperator between name parts.
    :type sep: str, optional

    :return: the generated name
    :rtype: str

    :raises ValueError: if ``param2`` is equal to ``param1``.
    """

    from .words import (adjectives, creatures)

    name = None
    entropy = max(entropy, 1)
    parts = (creatures, ) + (adjectives, ) * (entropy - 1)
    for _ in range(10):
        name_parts = [random.choice(p) for p in parts[::-1]]
        joined = sep.join(name_parts)
        if len(joined) <= max_length:
            name = joined
            break

    if not name:
        raise ValueError("Can't generate name under these conditions")

    return name


def random_string(length, charset=string.printable):
    return ''.join(random.choice(charset) for i in range(length))


def random_filename(length=(3, 50)):
    if hasattr(length, "__iter__"):
        mn, mx = length
        length = random.randrange(mn, mx+1)  # randrange does not include upper bound
    return random_string(length, charset=string.ascii_letters)


def random_buf(size):
    assert size < 5 * 2**20, "This is too big for a buffer (%s)" % size
    return random_string(size).encode("latin-1")


def perchance(probabilty):
    return random.random() <= probabilty
