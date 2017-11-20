import pytest
from easypy.resilience import retrying


def test_basic_retrying():

    runs = 0

    @retrying(3, sleep=0)
    def buggy():
        nonlocal runs
        runs += 1
        1/0

    with pytest.raises(ZeroDivisionError):
        buggy()

    assert runs == 4


def test_retrying_no_exception():

    runs = 0

    @retrying(3, sleep=0)
    def not_buggy(**kw):
        nonlocal runs
        runs += 1

    not_buggy()

    assert runs == 1


def test_retrying_with_param1():

    runs = 0

    @retrying(3, sleep=0)
    def buggy(retries_so_far):
        nonlocal runs
        runs = retries_so_far
        1/0

    with pytest.raises(ZeroDivisionError):
        buggy()

    assert runs == 4


def test_retrying_with_param2():

    runs = 0

    @retrying(3, sleep=0)
    def buggy(**kw):
        nonlocal runs
        runs = kw['retries_so_far']
        1/0

    with pytest.raises(ZeroDivisionError):
        buggy()

    assert runs == 4
