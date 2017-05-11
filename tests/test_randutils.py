import pytest
import random

from easypy.randutils import random_nice_name


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
