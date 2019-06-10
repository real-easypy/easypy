

def test_ziplog():
    from io import StringIO
    from easypy import ziplog
    from textwrap import dedent

    streams = dedent("""
    01:21:27                         -  2
    05:41:27                         -  4
    ;
    16255 15:08:52.554223|           -  5
    16155 19:08:52.554223|           - 11
    ;
    2018-04-01 04:48:11,811|         -  1
    2018-04-06 17:13:40,966          -  8
    ;
    2018-04-06T02:11:06+0200         -  3
    2018-04-07T02:11:06+0200         - 12
    ;
    2018-04-06 18:13:40,966          - 10
    2018-04-23 04:48:11,811|         - 14
    ;
    [2018/04/06 17:13:40.955356      -  7
    [2018/04/06 17:13:41.955356      -  9
    ;
    Apr 6 17:13:40                   -  6
    Apr 7 17:13:40                   - 13
    ;
    """)

    ziplog.YEAR = 2018
    ziplog.MONTH = 4
    ziplog.DAY = 6

    streams = [StringIO(line.lstrip()) for line in streams.split(";")]
    lines = ziplog.iter_zipped_logs(*streams, prefix="> ")
    prev = 0
    print()
    for line in lines:
        print(line, end="")
        cur = int(line.rpartition(" ")[-1])
        try:
            assert cur == prev + 1, "line %s is out of place" % cur
        except AssertionError:
            for line in lines:
                print(line, end="")
            raise
        prev = cur
