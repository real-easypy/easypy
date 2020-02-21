def test_resilience():
    from easypy.resilience import resilient

    @resilient(default=resilient.CAPTURE)
    def foo(a):
        return 1 / a

    ret = foo(1)
    assert ret == 1

    exc = foo(0)
    assert isinstance(exc, ZeroDivisionError)
