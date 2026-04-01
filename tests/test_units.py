import pytest

from easypy.units import byte, KiB, MiB, GiB, TiB, PiB, SECOND
from easypy.units import to_data_size, UnknownDataSizeError

def test_data_sizes():
    assert "{0!s}, {0!r}, {0:10text}".format(byte) == "byte, byte,     1 byte"
    assert "{0!s}, {0!r}, {0:10text}".format(1000 * byte) == "KB, KB,       1 KB"
    assert "{0!s}, {0!r}, {0:10text}".format(1020 * byte) == "1020bytes, 1020*bytes,  1020bytes"
    assert "{0!s}, {0!r}, {0:10text}".format(1024 * byte) == "KiB, KiB,      1 KiB"
    assert "{0!s}, {0!r}, {0:10text}".format(2**20 * byte) == "MiB, MiB,      1 MiB"
    assert "{0!s}, {0!r}, {0:10text}".format(2**21 * byte) == "2MiB, 2*MiB,       2MiB"
    assert "{0!s}, {0!r}, {0:10text}".format(2**21 * byte + 100) == "~2.0MiB, 2097252*bytes,    ~2.0MiB"
    assert "{0!s}, {0!r}, {0:10text}".format(2**41 * byte + 100) == "~2.0TiB, 2199023255652*bytes,    ~2.0TiB"


def test_durations():
    assert "{0!s}, {0!r}, {0:10text}".format(SECOND) == "1.0, 1s, no-time   "
    assert "{0!s}, {0!r}, {0:10text}".format(50 * SECOND) == "50.0, 50s, almost a minute"
    assert "{0!s}, {0!r}, {0:10text}".format(60 * SECOND) == "60.0, 01:00m, a minute  "
    assert "{0!s}, {0!r}, {0:10text}".format(60**2 * SECOND) == "3600.0, 01:00h, an hour   "
    assert "{0!s}, {0!r}, {0:10text}".format(25 * 60**2 * SECOND) == "90000.0, 1d, 01:00h, a day     "
    assert "{0!s}, {0!r}, {0:10text}".format(8 * 24 * 60**2 * SECOND) == "691200.0, 8d, 00:00h, 8 days    "
    assert "{0!s}, {0!r}, {0:10text}".format(32 * 24 * 60**2 * SECOND) == "2764800.0, 32d, 00:00h, 32 days   "
    assert "{0!s}, {0!r}, {0:10text}".format(400 * 24 * 60**2 * SECOND) == "34560000.0, 400d, 00:00h, 400 days  "


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


def test_to_data_size_edge_cases():
    assert to_data_size("0G") == 0
    assert to_data_size("1.5G") == 2 * GiB
    assert to_data_size("  10G  ") == 10 * GiB

    for invalid in ["k", "M", "GiB", "Ki", "", " "]:
        with pytest.raises(UnknownDataSizeError):
            to_data_size(invalid)


def test_to_data_size():
    assert to_data_size(10) == 10
    assert to_data_size(1.5) == 1.5
    assert to_data_size("10") == 10
    assert to_data_size("10k") == 10 * KiB
    assert to_data_size("10K") == 10 * KiB
    assert to_data_size("10m") == 10 * MiB
    assert to_data_size("10M") == 10 * MiB
    assert to_data_size("10g") == 10 * GiB
    assert to_data_size("10G") == 10 * GiB
    assert to_data_size("10t") == 10 * TiB
    assert to_data_size("10T") == 10 * TiB
    assert to_data_size("10p") == 10 * PiB
    assert to_data_size("10P") == 10 * PiB


def test_to_data_size_iec_suffixes():
    assert to_data_size("10Gi") == 10 * GiB
    assert to_data_size("10gi") == 10 * GiB
    assert to_data_size("10GiB") == 10 * GiB
    assert to_data_size("10gib") == 10 * GiB
    assert to_data_size("10Ki") == 10 * KiB
    assert to_data_size("10KiB") == 10 * KiB
    assert to_data_size("10Mi") == 10 * MiB
    assert to_data_size("10MiB") == 10 * MiB
    assert to_data_size("10Ti") == 10 * TiB
    assert to_data_size("10TiB") == 10 * TiB
    assert to_data_size("10Pi") == 10 * PiB
    assert to_data_size("10PiB") == 10 * PiB

    for invalid in ["10GB", "10Gb", "10gb", "10xyz", "abc"]:
        with pytest.raises(UnknownDataSizeError):
            to_data_size(invalid)
