import warnings
import pytest

from easypy.timing import iter_wait, wait, repeat, iter_wait_progress, Timer, TimeoutException
from easypy.bunch import Bunch


# Temporary tests for iter_wait() warnings
def test_wait_warning():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        wait(0.1, pred=lambda: True)
        wait(0.1)
        wait(0.1, pred=lambda: True, message='message')
        wait(0.1, pred=lambda: True, message=False)
        repeat(0.1, callback=lambda: True)

        # Only the first call should throw a DeprecationWarning
        assert len(w) == 1
        assert issubclass(w[-1].category, DeprecationWarning)
        assert "wait()" in str(w[-1].message)


def test_iter_wait_warning():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        for _ in iter_wait(0.1, pred=lambda: True):
            pass

        no_warn_iters = [
            iter_wait(0.1),
            iter_wait(0.1, pred=lambda: True, message='message'),
            iter_wait(0.1, pred=lambda: True, throw=False),
            iter_wait(0.1, pred=lambda: True, message=False)
        ]
        for i in no_warn_iters:
            for _ in i:
                pass

        # Only the first call should throw a DeprecationWarning
        assert len(w) == 1
        assert issubclass(w[-1].category, DeprecationWarning)
        assert "wait()" in str(w[-1].message)


def test_iter_wait_progress_inbetween_sleep():
    data = Bunch(a=3)

    def get():
        data.a -= 1
        return data.a

    sleep = .07
    g = iter_wait_progress(get, advance_timeout=10, sleep=sleep)

    # first iteration should be immediate
    t = Timer()
    next(g)
    assert t.duration < sleep

    # subsequent iteration should be at least 'sleep' long
    next(g)
    assert t.duration >= sleep

    for state in g:
        pass
    assert state.finished is True


def test_iter_wait_progress_total_timeout():
    data = Bunch(a=1000)

    def get():
        data.a -= 1
        return data.a

    with pytest.raises(TimeoutException) as exc:
        for state in iter_wait_progress(get, advance_timeout=1, sleep=.05, total_timeout=.1):
            pass
    assert exc.value.message.startswith("advanced but failed to finish")
