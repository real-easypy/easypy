from easypy.units import byte


def test_data_sizes():
    assert "{0!s}, {0!r}, {0:10text}".format(byte) == "byte, byte,     1 byte"
    assert "{0!s}, {0!r}, {0:10text}".format(1000 * byte) == "KB, KB,       1 KB"
    assert "{0!s}, {0!r}, {0:10text}".format(1020 * byte) == "1020bytes, 1020*bytes,  1020bytes"
    assert "{0!s}, {0!r}, {0:10text}".format(1024 * byte) == "KiB, KiB,      1 KiB"
    assert "{0!s}, {0!r}, {0:10text}".format(2**20 * byte) == "MiB, MiB,      1 MiB"
    assert "{0!s}, {0!r}, {0:10text}".format(2**21 * byte) == "2MiB, 2*MiB,       2MiB"
    assert "{0!s}, {0!r}, {0:10text}".format(2**21 * byte + 100) == "~2.0MiB, 2097252*bytes,    ~2.0MiB"
    assert "{0!s}, {0!r}, {0:10text}".format(2**41 * byte + 100) == "~2.0TiB, 2199023255652*bytes,    ~2.0TiB"
