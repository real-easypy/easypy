import pytest
import random

from easypy.random import random_nice_name, random_filename


def test_random_nice_name():
    for _ in range(20):
        length = random.randint(64, 85)
        entropy = random.randint(1, 3)
        sep = random.choice(['_', '..'])
        name = random_nice_name(max_length=length, entropy=entropy, sep=sep)
        assert len(name) <= length


def test_random_nice_name_raises():
    with pytest.raises(ValueError):
        random_nice_name(max_length=10, entropy=3)


def test_random_filename():
    fn = random_filename(10)
    assert len(fn) == 10

    fn = random_filename((10, 11))
    assert 10 <= len(fn) <= 11

    fn = random_filename((11, 11))
    assert len(fn) == 11
