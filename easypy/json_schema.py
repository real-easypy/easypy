import inspect


_BUILDERS_BY_TYPE = {}
_BUILDERS_BY_TYPE[str] = lambda typ: dict(type='string')
_BUILDERS_BY_TYPE[int] = lambda typ: dict(type='number', multipleOf=1)
_BUILDERS_BY_TYPE[float] = lambda typ: dict(type='number')
_BUILDERS_BY_TYPE[bool] = lambda typ: dict(type='boolean')
_BUILDERS_BY_TYPE[type(None)] = lambda typ: dict(type='null')

_BUILDERS_BY_TYPE[list] = lambda typ: dict(type='array')
_BUILDERS_BY_TYPE[dict] = lambda typ: dict(type='object')


def as_json_schema(typ):
    """
    Return the JSON schema that matches serialized objects of the type.
    """
    try:
        builder = typ._as_json_schema_
    except AttributeError:
        pass
    else:
        return builder()
    for cls in inspect.getmro(typ):
        try:
            builder = _BUILDERS_BY_TYPE[cls]
        except KeyError:
            pass
        else:
            return builder(typ)
    return {}  # accepts any valid JSON
