from easypy.units import byte, KiB


def test_data_sizes():
    assert "{0!s}, {0!r}, {0:10text}".format(byte) == "byte, byte,     1 byte"
    assert "{0!s}, {0!r}, {0:10text}".format(1000 * byte) == "KB, KB,       1 KB"
    assert "{0!s}, {0!r}, {0:10text}".format(1020 * byte) == "1020bytes, 1020*bytes,  1020bytes"
    assert "{0!s}, {0!r}, {0:10text}".format(1024 * byte) == "KiB, KiB,      1 KiB"
    assert "{0!s}, {0!r}, {0:10text}".format(2**20 * byte) == "MiB, MiB,      1 MiB"
    assert "{0!s}, {0!r}, {0:10text}".format(2**21 * byte) == "2MiB, 2*MiB,       2MiB"
    assert "{0!s}, {0!r}, {0:10text}".format(2**21 * byte + 100) == "~2.0MiB, 2097252*bytes,    ~2.0MiB"
    assert "{0!s}, {0!r}, {0:10text}".format(2**41 * byte + 100) == "~2.0TiB, 2199023255652*bytes,    ~2.0TiB"


def test_operators():

    assert (byte * 1024) == KiB
    assert KiB / 1024 == byte
    assert KiB / KiB == 1
    assert KiB / 7 == 146.28571428571428
    assert KiB // 7 == 146
    assert 2050 // KiB == (2 * byte)

    # check that __r*__ overloads are used when the unit doesn't support the right-hand operand
    class Foo():
        def __rfloordiv__(self, div):
            return self

    foo = Foo()

    assert KiB // foo is foo
