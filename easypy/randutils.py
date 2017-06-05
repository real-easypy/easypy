import os
import string
import itertools
import random
import bisect
import collections
from random import choice, sample

choose = choice


def interpolate(val, mn, mx):
    return (val*(mx-mn) + mn)


def clamp(val, mn, mx):
    return max(mn, min(mx, val))


class XRandom(random.Random):

    def choose_weighted(self, *weighted_choices):
        choices, weights = zip(*weighted_choices)
        cumdist = list(itertools.accumulate(weights))
        x = self.random() * cumdist[-1]
        return choices[bisect.bisect(cumdist, x)]

    def get_size(self, lo, hi, exp=2):
        return int(interpolate(self.random()**exp, lo, hi))

    def get_chunks(self, offset, end, block_size_range, shuffle=False):
        if shuffle:
            chunks = list(self.get_chunks(offset, end, block_size_range))
            self.shuffle(chunks)
            yield from chunks
            return

        total_size = end - offset
        while total_size:
            size = self.get_size(*block_size_range)
            size = clamp(size, 1, total_size)
            if size:
                yield offset, size
            total_size -= size
            offset += size


def random_nice_name(max_length=64, entropy=2, sep='-'):
    """Generates a nice random name from the dictionaries in words
    Args:
        max_length (int, optional): Max length for the name.
        entropy (int, optional): How unique th name will be, currently
            entropy - 1 adjectives are joined with one noun.
        sep (str, optional: Seperator between name parts.

    Returns:
        string: The generated name.

    Raises:
        ValueError: If `param2` is equal to `param1`.
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
