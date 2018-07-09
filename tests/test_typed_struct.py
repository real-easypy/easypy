import pytest

from easypy.bunch import Bunch

import easypy.typed_struct as ts


def test_typed_struct_validations():
    class OutOfRangeError(Exception):
        pass

    class Foo(ts.TypedStruct):
        a = ts.Field(int, default=20)
        a.add_validation(range(10, 30).__contains__, OutOfRangeError)
        b = ts.Field(str)

    # Valid construction
    Foo(b='hi')
    Foo(a=14, b='hello')

    # Invalid construction
    with pytest.raises(ts.NotFields) as exc:
        Foo(b='', c=10)
    assert exc.value.message == 'c are not fields of Foo'

    with pytest.raises(ts.MissingField) as exc:
        Foo()
    assert exc.value.message == 'Foo is missing mandatory field b'

    with pytest.raises(ts.FieldTypeMismatch):
        Foo(a=20.0, b='')

    with pytest.raises(OutOfRangeError):
        Foo(a=100, b='')

    foo = Foo(b='')

    # Valid attribute assignments
    foo.a = 15
    foo.b = 'hello'

    # Invalid attribute assignments
    with pytest.raises(ts.NotAField) as exc:
        foo.c = 10
    assert exc.value.message == 'c is not a field of Foo'

    with pytest.raises(ts.FieldTypeMismatch):
        foo.b = []

    with pytest.raises(OutOfRangeError):
        foo.a = 1000

    assert foo.a == 15
    assert foo.b == 'hello'

    # Valid key assignments
    foo['a'] = 25
    foo['b'] = 'hi'

    # Invalid key assignments
    with pytest.raises(ts.NotAKey) as exc:
        foo['c'] = 10
    assert exc.value.message == 'c is not a key of Foo'

    with pytest.raises(ts.FieldTypeMismatch):
        foo['b'] = []

    with pytest.raises(OutOfRangeError):
        foo['a'] = 1000

    assert foo['a'] == 25
    assert foo['b'] == 'hi'


def test_typed_struct_nesting():
    class Foo(ts.TypedStruct):
        a = ts.Field(int, default=0)
        b = ts.Field(int, default=1)

    class Bar(ts.TypedStruct):
        c = ts.Field(int, default=2)
        d = ts.Field(int)

    class Baz(ts.TypedStruct):
        foo = ts.Field(Foo, default=Foo(a=10))
        foo2 = Foo
        bar = Bar

    baz = Baz(bar=dict(d=3))
    assert baz.foo == Foo(a=10, b=1)
    assert baz.foo2 == Foo()
    assert baz.bar == Bar(c=2, d=3)

    # Very important - nested fields are modifiable so if the default is used
    # they must not be shared!
    assert baz.foo is baz.foo
    assert baz.foo is not Baz(bar=dict(d=3)).foo


def test_typed_struct_comparison():
    class Foo(ts.TypedStruct):
        a = int
        b = int

    class Bar(ts.TypedStruct):
        a = int
        b = int

    assert Foo(a=1, b=2) == Foo(a=1, b=2)
    assert Foo(a=1, b=2) != Foo(a=1, b=3)
    assert Foo(a=1, b=2) != Bar(a=1, b=2)

    assert Foo(a=1, b=2).to_dict() == Bar(a=1, b=2).to_dict()
    assert Foo(a=1, b=2).to_bunch() == Bar(a=1, b=2).to_bunch()


@pytest.mark.parametrize("use_full_syntax", [True, False])
def test_typed_struct_list_fields(use_full_syntax):
    class Foo(ts.TypedStruct):
        a = int

    if use_full_syntax:
        class Bar(ts.TypedStruct):
            foos = ts.Field([Foo])
            nums = ts.Field([int])
    else:
        class Bar(ts.TypedStruct):
            foos = [Foo]
            nums = [int]

    # C'tor
    bar = Bar(foos=[Foo(a=1), Foo(a=2)], nums=[1, 2, 3])
    assert bar.foos == [Foo(a=1), Foo(a=2)]
    assert bar.nums == [1, 2, 3]

    # Default
    bar = Bar()
    assert bar.foos == []
    assert bar.nums == []

    # append()
    bar.foos.append(Foo(a=1))
    bar.foos.append(dict(a=2))
    with pytest.raises(ts.FieldTypeMismatch):
        bar.foos.append(3)
    assert bar.foos == [Foo(a=1), Foo(a=2)]

    bar.nums.append(1)
    with pytest.raises(ts.FieldTypeMismatch):
        bar.nums.append('2')
    assert bar.nums == [1]

    # __setitem__()
    bar.foos[0] = Foo(a=4)
    bar.foos[1] = dict(a=5)
    with pytest.raises(ts.FieldTypeMismatch):
        bar.foos[0] = 6
    assert bar.foos == [Foo(a=4), Foo(a=5)]

    bar.nums[0] = 3
    assert bar.nums == [3]

    # insert()
    bar.foos.insert(0, Foo(a=7))
    bar.foos.insert(2, dict(a=8))
    with pytest.raises(ts.FieldTypeMismatch):
        bar.foos.insert(1, 9)
    assert bar.foos == [Foo(a=7), Foo(a=4), Foo(a=8), Foo(a=5)]

    bar.nums.insert(0, 4)
    assert bar.nums == [4, 3]

    # Assignment
    bar.foos = [Foo(a=10), dict(a=11)]
    with pytest.raises(ts.FieldTypeMismatch):
        bar.foos = [12]
    assert bar.foos == [Foo(a=10), Foo(a=11)]

    bar.nums = [5]
    assert bar.nums == [5]

    # __iadd__()
    bar.foos += [Foo(a=13), dict(a=14)]
    with pytest.raises(ts.FieldTypeMismatch):
        bar.foos += [15]
    assert bar.foos == [Foo(a=10), Foo(a=11), Foo(a=13), Foo(a=14)]

    bar.nums += [6]
    assert bar.nums == [5, 6]

    # extend()
    bar.foos.extend(iter([Foo(a=16), dict(a=17)]))
    with pytest.raises(ts.FieldTypeMismatch):
        bar.foos.extend(range(18))
    assert bar.foos == [Foo(a=10), Foo(a=11), Foo(a=13), Foo(a=14), Foo(a=16), Foo(a=17)]

    bar.nums.extend(iter([7]))
    assert bar.nums == [5, 6, 7]

    # Make sure that bar.foos and bar.nums are still a ListCollection
    assert bar.foos.M.a.C == [10, 11, 13, 14, 16, 17]
    assert bar.nums.M.call(lambda x: x * 10).C == [50, 60, 70]


@pytest.mark.parametrize("use_full_syntax", [True, False])
def test_typed_struct_dict_fields(use_full_syntax):
    class Foo(ts.TypedStruct):
        a = int

    if use_full_syntax:
        class Bar(ts.TypedStruct):
            foos = ts.Field({int: Foo})
            nums = ts.Field({int: int})
    else:
        class Bar(ts.TypedStruct):
            foos = {int: Foo}
            nums = {int: int}

    # C'tor
    bar = Bar(foos={1: Foo(a=2)}, nums={3: 4})
    assert bar.foos == {1: Foo(a=2)}
    assert bar.nums == {3: 4}

    # Default
    bar = Bar()
    assert bar.foos == {}
    assert bar.nums == {}

    # __setitem__()

    bar.foos[1] = Foo(a=2)
    bar.foos[3] = dict(a=4)
    with pytest.raises(ts.FieldTypeMismatch):
        bar.foos[6] = 7
    with pytest.raises(ts.FieldKeyTypeMismatch):
        bar.foos['8'] = Foo(a=9)
    assert bar.foos == {1: Foo(a=2), 3: Foo(a=4)}

    bar.nums[1] = 2
    with pytest.raises(ts.FieldTypeMismatch):
        bar.nums[3] = '4'
    with pytest.raises(ts.FieldKeyTypeMismatch):
        bar.nums['5'] = 6
    assert bar.nums == {1: 2}

    # Assignment

    bar.foos = {5: dict(a=6), 7: Foo(a=8)}
    with pytest.raises(ts.FieldTypeMismatch):
        bar.foos = {9: 10}
    with pytest.raises(ts.FieldKeyTypeMismatch):
        bar.foos = {'11': Foo(a=12)}
    assert bar.foos == {5: Foo(a=6), 7: Foo(a=8)}

    bar.nums = {7: 8}
    with pytest.raises(ts.FieldTypeMismatch):
        bar.nums = {9: '10'}
    with pytest.raises(ts.FieldKeyTypeMismatch):
        bar.nums = {'11': 12}
    assert bar.nums == {7: 8}

    # setdefault()

    assert bar.foos.setdefault(5, Foo(a=13)).a == 6  # from the assignments tests
    assert bar.foos.setdefault(14, Foo(a=15)).a == 15
    assert bar.foos.setdefault(16, dict(a=17)).a == 17
    with pytest.raises(ts.FieldTypeMismatch):
        bar.foos.setdefault(18, 19)
    with pytest.raises(ts.FieldKeyTypeMismatch):
        bar.foos.setdefault('20', Foo(a=21))
    assert bar.foos == {5: Foo(a=6), 7: Foo(a=8), 14: Foo(a=15), 16: Foo(a=17)}

    assert bar.nums.setdefault(7, 13) == 8  # NOTE: from the assignments tests
    assert bar.nums.setdefault(14, 15) == 15
    with pytest.raises(ts.FieldTypeMismatch):
        bar.nums.setdefault(16, '17')
    with pytest.raises(ts.FieldKeyTypeMismatch):
        bar.nums.setdefault('18', 19)
    assert bar.nums == {7: 8, 14: 15}

    bar.foos.clear()  # it got too big

    # update()

    # NOTE: I can't test the .update(bla=...) syntax, because I only accept
    # ints here.

    bar.foos.update({22: Foo(a=23), 24: dict(a=25)})
    with pytest.raises(ts.FieldTypeMismatch):
        bar.foos.update({26: 27})
    with pytest.raises(ts.FieldKeyTypeMismatch):
        bar.foos.update({'28': Foo(a=29)})
    assert bar.foos == {22: Foo(a=23), 24: Foo(a=25)}

    bar.nums.update({20: 21})
    with pytest.raises(ts.FieldTypeMismatch):
        bar.nums.update({22: '23'})
    with pytest.raises(ts.FieldKeyTypeMismatch):
        bar.nums.update({'24': 25})
    assert bar.nums == {7: 8, 14: 15, 20: 21}


@pytest.mark.parametrize("use_full_syntax", [True, False])
def test_typed_struct_bunch_fields(use_full_syntax):
    if use_full_syntax:
        class Bar(ts.TypedStruct):
            nums = ts.Field({str: int})
    else:
        class Bar(ts.TypedStruct):
            nums = {str: int}

    # NOTE: TypedBunch shares most things with TypedDict, so I'm only doing a quick sanity test

    bar = Bar(nums=Bunch(a=0))
    assert bar.nums.a == 0

    bar.nums.a = 1
    bar.nums['b'] = 2
    bar.nums.update(c=3)
    with pytest.raises(ts.FieldTypeMismatch):
        bar.nums.d = '4'
    with pytest.raises(ts.FieldKeyTypeMismatch):
        bar.nums[5] = 6

    assert bar.nums == Bunch(a=1, b=2, c=3)


def test_typed_struct_field_defaults():
    class Foo(ts.TypedStruct):
        x = int

    class Bar(ts.TypedStruct):
        a = ts.Field(int, default=1)
        b = ts.Field([int], default=[2, 3])
        c = ts.Field(Foo, default=Foo(x=4))
        d = ts.Field(Foo, default=dict(x=5))
        e = ts.Field([Foo], default=[Foo(x=6), dict(x=7)])
        f = ts.Field({str: Foo}, default=dict(g=Foo(x=8)))

    bar1 = Bar()
    bar2 = Bar()

    assert bar1.a == bar2.a == 1

    assert bar1.b == bar2.b == [2, 3]
    assert bar1.b is not bar2.b

    assert bar1.c == bar2.c == Foo(x=4)
    assert bar1.c is not bar2.c

    assert bar1.d == bar2.d == Foo(x=5)
    assert bar1.d is not bar2.d

    assert bar1.e == bar2.e == [Foo(x=6), Foo(x=7)]
    assert bar1.e is not bar2.e
    assert bar1.e[0] is not bar2.e[0]
    assert bar1.e[1] is not bar2.e[1]

    assert bar1.f == bar2.f == Bunch(g=Foo(x=8))
    assert bar1.f is not bar2.f
    assert bar1.f.g is not bar2.f.g


def test_typed_struct_metadata():
    class Foo(ts.TypedStruct):
        a = ts.Field(int, meta=dict(format='%d'))
        b = ts.Field(int, meta=dict(format='%03d'))

    class Bar(ts.TypedStruct):
        c = ts.Field(str, meta=dict(format='%s'))
        d = ts.Field(str, meta=dict(format='"%s"'))

    def format_object(obj):
        def gen():
            for k, v in obj.items():
                field = getattr(type(obj), k)
                formatted = field.meta.format % (v,)
                yield '%s=%s' % (k, formatted)
        return '%s[%s]' % (type(obj).__name__, ', '.join(gen()))

    assert format_object(Foo(a=1, b=2)) == 'Foo[a=1, b=002]'
    assert format_object(Bar(c='3', d='4')) == 'Bar[c=3, d="4"]'


def test_typed_struct_convert():
    class Foo(ts.TypedStruct):
        a = ts.Field(int)
        a.add_conversion(str, int)
        a.add_conversion(float, int)

    assert Foo(a=1).a == 1
    assert Foo(a='2').a == 2
    assert Foo(a=3.4).a == 3
    with pytest.raises(ts.FieldTypeMismatch):
        # Don't know how to convert complex
        Foo(a=5 + 6j).a


@pytest.mark.parametrize("use_helper_methods", [True, False])
def test_typed_struct_preprocess(use_helper_methods):
    class Error1(Exception):
        pass

    class Error2(Exception):
        pass

    if use_helper_methods:
        class Foo(ts.TypedStruct):
            a = ts.Field(int)

            a.add_validation(lambda n: n % 2 == 0, Error1, 'not an even number')
            a.add_validation(range(10, 31).__contains__, Error2, 'out of range([10, 30])')

            a.add_conversion(str, int)
            a.add_conversion(float, int)
    else:
        def preprocess(obj):
            if isinstance(obj, str) or isinstance(obj, float):
                obj = int(obj)

            if isinstance(obj, int):
                if obj % 2 == 1:
                    raise Error1('not an even number')
                if obj not in range(10, 31):
                    raise Error2('out of range([10, 30])')

            return obj

        class Foo(ts.TypedStruct):
            a = ts.Field(int, preprocess=preprocess)

    assert Foo(a=12).a == 12
    assert Foo(a='12').a == 12
    assert Foo(a=12.0).a == 12

    with pytest.raises(ts.FieldTypeMismatch):
        Foo(a=[])

    # Raising errors on validations:

    with pytest.raises(Error1) as exc:
        Foo(a=13)
    assert exc.value.args[0] == 'not an even number'

    with pytest.raises(Error2) as exc:
        Foo(a=32)
    assert exc.value.args[0] == 'out of range([10, 30])'

    # Raising errors on validations even after conversions:

    with pytest.raises(Error1):
        Foo(a=13.0)

    with pytest.raises(Error2):
        Foo(a='32')


def test_typed_struct_collection_type_verification():
    """
    To prevent things like this:

        In [1]: from easypy import typed_struct as ts

        In [2]: class Foo(ts.TypedStruct):
           ...:     bars = [str]
           ...:

        In [3]: Foo(bars='hello')
        Out[3]:
        Foo(bars=['h',
                  'e',
                  'l',
                  'l',
                  'o'])
    """

    class Foo(ts.TypedStruct):
        a = ts.Field([str])
        b = ts.Field({str: str})
        c = ts.Field({int: str})

    # Test list:

    assert Foo(a=['1']).a == ['1']
    with pytest.raises(ts.FieldCollectionTypeMismatch):
        Foo(a='2')
    with pytest.raises(ts.FieldCollectionTypeMismatch):
        Foo(a={'3': '4'})

    # Test dict:

    assert Foo(b={'5': '6'}).b == {'5': '6'}
    with pytest.raises(ts.FieldCollectionTypeMismatch):
        Foo(b='8')
    with pytest.raises(ts.FieldCollectionTypeMismatch):
        Foo(b=['9'])

    # Test Bunch:

    assert Foo(c={10: '11'}).c == {10: '11'}
    with pytest.raises(ts.FieldCollectionTypeMismatch):
        Foo(c='13')
    with pytest.raises(ts.FieldCollectionTypeMismatch):
        Foo(c=['14'])


def test_typed_struct_auto_field_wrapping_dsl():
    class Foo(ts.TypedStruct):
        a = int
        a.default = 1
        b = float
        b.convertible_from(str)

    assert Foo(b='2.3').to_dict() == dict(a=1, b=2.3)


def test_typed_struct_inheritance():
    class Foo(ts.TypedStruct):
        a = int
        a.default = 1

        b = int
        b.default = 2

    class Bar(Foo):
        a.default = 3

        c = int
        c.default = 4

    assert Foo().to_dict() == dict(a=1, b=2)
    assert Bar().to_dict() == dict(a=3, b=2, c=4)


def test_typed_struct_repr():
    class Foo(ts.TypedStruct):
        a = int
        b = float
        b.repr = False
        c = str
        c.repr = '`{}`'.format

    assert repr(Foo(a=1, b=2.0, c='3')) == 'Foo(a=1, c=`3`)'


def test_typed_struct_hash():
    class Foo(ts.TypedStruct):
        a = int
        b = int
        b.hash = False
        c = int
        c.hash = lambda n: n % 2


    assert hash(Foo(a=1, b=2, c=3)) == hash(Foo(a=1, b=2, c=3))
    assert hash(Foo(a=1, b=2, c=3)) != hash(Foo(a=2, b=2, c=3))
    assert hash(Foo(a=1, b=2, c=3)) == hash(Foo(a=1, b=1, c=3)), 'b should not be part of the hash'
    assert hash(Foo(a=1, b=2, c=3)) == hash(Foo(a=1, b=2, c=1)), 'Only the parity of c should be in the hash'
    assert hash(Foo(a=1, b=2, c=3)) != hash(Foo(a=1, b=2, c=2)), 'c with different parity should generate different hash'

    class Bar(ts.TypedStruct):
        a = int
        b = int
        b.hash = False
        c = int
        c.hash = lambda n: n % 2

    assert hash(Foo(a=1, b=2, c=3)) != hash(Bar(a=1, b=2, c=3)), \
        'different classes with same fields should have different hahses'
