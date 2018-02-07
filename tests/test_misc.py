def test_tokens():
    from easypy.tokens import AUTO, if_auto, MAX

    def foo(p=AUTO):
        return if_auto(p, 100)

    assert foo() == 100
    assert foo(5) == 5
    assert foo(MAX) == MAX
