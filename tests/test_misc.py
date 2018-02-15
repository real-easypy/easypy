def test_tokens():
    from easypy.tokens import AUTO, if_auto, MAX

    def foo(p=AUTO):
        return if_auto(p, 100)

    assert foo() == 100
    assert foo(5) == 5
    assert foo(MAX) == MAX

    assert MAX == "MAX"
    assert MAX == "<MAX>"
    assert MAX == "max"
    assert MAX == "<max>"

    d = {AUTO: AUTO, MAX: MAX}
    assert d[AUTO] == AUTO
    assert d[MAX] == MAX
    assert d['<MAX>'] is MAX
    assert 'AUTO' not in d
