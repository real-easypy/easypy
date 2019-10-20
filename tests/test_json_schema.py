from easypy.json_schema import as_json_schema
from easypy.typed_struct import TypedStruct


def test_json_schema_primitives():
    assert as_json_schema(str) == {'type': 'string'}
    assert as_json_schema(int) == {'type': 'number', 'multipleOf': 1}
    assert as_json_schema(float) == {'type': 'number'}
    assert as_json_schema(bool) == {'type': 'boolean'}
    assert as_json_schema(type(None)) == {'type': 'null'}
    assert as_json_schema(list) == {'type': 'array'}
    assert as_json_schema(dict) == {'type': 'object'}


def test_json_schema_typed_struct():
    class Foo(TypedStruct):
        a = str
        a.default = 'one'
        b = int
        b.default = 2
        c = float
        d = bool

    class Bar(TypedStruct):
        e = [Foo]
        f = {str: Foo}
        f.default = {}

    assert Bar._as_json_schema_() == dict(
        type='object',
        additionalProperties=False,
        required=['e'],
        properties=dict(
            e=dict(
                type='array',
                items=dict(
                    type='object',
                    additionalProperties=False,
                    required=['c', 'd'],
                    properties=dict(
                        a=dict(type='string'),
                        b=dict(type='number', multipleOf=1),
                        c=dict(type='number'),
                        d=dict(type='boolean'),
                    ),
                ),
            ),
            f=dict(
                type='object',
                additionalProperties=dict(
                    type='object',
                    additionalProperties=False,
                    required=['c', 'd'],
                    properties=dict(
                        a=dict(type='string'),
                        b=dict(type='number', multipleOf=1),
                        c=dict(type='number'),
                        d=dict(type='boolean'),
                    ),
                ),
            ),
        ),
    )
