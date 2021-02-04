from easypy.humanize import from_hexdump, hexdump, IndentableTextBuffer, format_table, easy_repr, Histogram


_SAMPLE_DATA = b'J\x9c\xe8Z!\xc2\xe6\x8b\xa0\x01\xcb\xc3.x]\x11\x9bsC\x1c\xb2\xcd\xb3\x9eM\xf7\x13`\xc8\xce\xf8g1H&\xe2\x9b'     \
    b'\xd1\xa8\xfd\x14\x08U\x175\xc7\x03q\xac\xda\xe6)q}}T44\x9e\xb5;\xf1.\xf6*\x16\xba\xe0~m\x96o\xb8\xa4Tl\x96\x8a\xc7'    \
    b'\x9a\xc9\xc4\xf2\xb1\x9e\x13\x0b\xe2i\xc6\xd8\x92\xde\xfabn6\xea\xf5_y>\x15\xc5\xd5\xa0\x05\xbd\xea\xb8\xba\x80+P'     \
    b'\xa7\xd8\xad\xbf\x91<\xca\xc5\x94\xe6\xfc-\xab4ABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABAB'     \
    b'ABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABA'     \
    b'BABABABABABABABABABABABABABABABABABABABABCABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABAB'    \
    b'ABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABA'     \
    b'BABABABABABABABABABABABABABABABABABABABABAB\xdc)n/\x9aNy\x9f\x03\xc7j\x14\x08\x1a\x08\x91@\xad\xac\xa9(\x1a\x8b\x9f'   \
    b'\x81\xb0us\x87\x9e4\xf9\x991w39\xd0\x98XokH\xa6\xc9Rv\xbc\xac\x90;\xac\x83\xc8\xba`V\xa9\xc3u\xb2\xccV\x9d\x06\xb3'    \
    b'\xf0\x1e\xb4K\x10\x9c\x83\xdc\xe7\xcb\x0c\x9a\x8c\x80\x010\x8ca\xf85Z\x9c'


def test_hexdump_functions():
    assert from_hexdump(hexdump(_SAMPLE_DATA)) == _SAMPLE_DATA
    assert from_hexdump(hexdump(_SAMPLE_DATA, 24, 2)) == _SAMPLE_DATA
    assert from_hexdump(hexdump(_SAMPLE_DATA, 16, 1, False)) == _SAMPLE_DATA
    assert from_hexdump(hexdump(_SAMPLE_DATA, 4, 4)) == _SAMPLE_DATA

    assert _SAMPLE_DATA.decode("hexdump_24_2") == hexdump(_SAMPLE_DATA, 24, 2)
    assert hexdump(_SAMPLE_DATA, 24, 2).encode("hexdump") == _SAMPLE_DATA


def test_indentable_text_buffer():
    from io import StringIO

    buff = IndentableTextBuffer("Exc")
    buff.write("a")
    buff.write("b")
    with buff.indent("Header2"):
        buff.write(hexdump(_SAMPLE_DATA, 24, 8))
    buff.write("hello")
    buff.write("world")
    with buff.indent("Header2"):
        # buff.write(format_in_columns([str(i) for i in range(100)], 50))
        with buff.indent("This should be pruned away"):
            with buff.indent("This should be pruned away"):
                pass
        with buff.indent("Header3"):
            buff.write("text3")
        buff.write("text2")

    f = StringIO()
    buff.render(prune=True, textual=True, width=120, file=f)
    assert open("tests/indentable_buffer1.txt", "r").read() == f.getvalue()

    f = StringIO()
    buff.render(prune=True, textual=False, width=40, overflow="ignore", file=f)
    assert open("tests/indentable_buffer2.txt", "r").read() == f.getvalue()

    f = StringIO()
    buff.render(prune=True, textual=False, width=40, edges=False, file=f)
    assert open("tests/indentable_buffer3.txt", "r").read() == f.getvalue()


def test_format_table_with_titles():
    table = [
        'abc',
        range(3),
        [None, True, False],
        [dict(x='x'), b'bytes', 'string']
    ]

    output = (
        "a         |b       |c     \n"
        "--------------------------\n"
        "         0|       1|     2\n"
        "None      |True    |False \n"
        "{'x': 'x'}|b'bytes'|string\n")

    assert output == format_table(table)


def test_format_table_without_titles():
    table = [
        'abc',
        range(3),
        [None, True, False],
        [dict(x='x'), b'bytes', 'string']
    ]

    output = (
        "a         |b       |c     \n"
        "         0|       1|     2\n"
        "None      |True    |False \n"
        "{'x': 'x'}|b'bytes'|string\n")

    assert output == format_table(table, titles=False)


def test_easy_repr():
    @easy_repr('a', 'b', 'c')
    class Class1:
        def __init__(self, a, b, c, d):
            self.a = a
            self.b = b
            self.c = c
            self.d = d
    a = Class1('a', 'b', 1, 2)
    assert repr(a) == "<Class1: a='a', b='b', c=1>"

    # change order
    @easy_repr('c', 'a', 'd')
    class Class2:
        def __init__(self, a, b, c, d):
            self.a = a
            self.b = b
            self.c = c
            self.d = d
    a = Class2('a', 'b', 1, 2)
    assert repr(a) == "<Class2: c=1, a='a', d=2>"

    try:
        @easy_repr()
        class Class3:
            def __init__(self, a, b, c, d):
                self.a = a
                self.b = b
                self.c = c
                self.d = d
    except AssertionError:
        pass
    else:
        assert False, 'easy_repr with no attributes should not be allowed'


def test_histogram():
    import functools
    from io import StringIO

    bins = [-100, 25, 100, -50]

    # sorted: [-100, -50, 25, 100]
    bins_sorted = sorted(bins)
    final_values = [1, 1, 2, 1, 1]
    percentages = [16.67, 16.67, 33.33, 16.67, 16.67]
    bars = [15, 15, 30, 15, 15]

    hist = Histogram(bins)
    hist.increment(1)
    assert hist.get_values() == [0, 0, 1, 0, 0]
    hist.increment(0)
    assert hist.get_values() == [0, 0, 2, 0, 0]
    hist.increment(25)
    assert hist.get_values() == [0, 0, 2, 1, 0]
    hist.increment(-250)
    assert hist.get_values() == [1, 0, 2, 1, 0]
    hist.increment(-60)
    assert hist.get_values() == [1, 1, 2, 1, 0]
    hist.increment(1124)
    assert hist.get_values() == final_values

    output = StringIO()

    def writeline_callback(output, fmt, *args):
        output.write(fmt % args + '\n')

    hist.show_graph(functools.partial(writeline_callback, output), 30)
    output.seek(0)
    lines = output.read().splitlines()
    assert len(lines) == len(final_values)
    for i, line in enumerate(lines):
        current_bins, percent, bar = line.split('|')
        if i == 0:
            assert '<' in current_bins
            assert current_bins.split('<')[1].strip() == str(bins_sorted[0])
        elif i == len(lines) - 1:
            assert '<' in current_bins
            assert current_bins.split('<')[0].strip() == str(bins_sorted[-1])
        else:
            middle = len(current_bins) // 2
            assert current_bins[middle] == '-'
            assert int(current_bins[:middle]) == bins_sorted[i - 1]
            assert int(current_bins[middle + 1:]) == bins_sorted[i]

        assert float(percent.strip(' %')) == percentages[i]
        assert bar.strip() == '-' * bars[i]
