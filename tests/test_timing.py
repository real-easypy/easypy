import warnings
import pytest

from easypy.timing import iter_wait, wait, repeat


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
