import re
import time
from datetime import datetime
from queue import PriorityQueue
from .colors import uncolored


TIMESTAMP_PATTERN = "%Y-%m-%d %H:%M:%S"


def to_timestamp(t):
    return "-".center(19) if t is None else time.strftime(TIMESTAMP_PATTERN, time.localtime(t))


YEAR = time.strftime("%Y")
MONTH = time.strftime("%m")
DAY = time.strftime("%d")

TIMESTAMP_GETTERS = [

    # 01:21:27
    (re.compile(r"^(\d+:\d+:\d+)"),
     lambda ts: time.mktime(time.strptime("%s-%s-%s %s" % (YEAR, MONTH, DAY, ts), "%Y-%m-%d %H:%M:%S"))),

    # Apr 6 17:13:40
    (re.compile(r"^(\w{3} +\d+ +\d+:\d+:\d+)"),
     lambda ts: time.mktime(time.strptime("%s %s" % (YEAR, ts), "%Y %b %d %H:%M:%S"))),

    # 2018-12-15T02:11:06+0200
    (re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+\d{4})"),
     lambda ts: datetime.strptime(ts, '%Y-%m-%dT%H:%M:%S%z').timestamp()),

    # 2018-12-15T02:11:06.123456+02:00
    # 2019-10-09T10:58:45,929228489+03:00
    (re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})[.,](\d{6})\d*(\+\d{2}):(\d{2})"),
     lambda *args: datetime.strptime("{}.{}{}{}".format(*args), '%Y-%m-%dT%H:%M:%S.%f%z').timestamp()),

    # 2018-04-06 17:13:40,955
    # 2018-04-23 04:48:11,811|
    (re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d{3})[| ]"),
     lambda ts, ms: time.mktime(time.strptime(ts, "%Y-%m-%d %H:%M:%S")) + float(ms) / 1000),

    # 2018-04-06 17:13:40
    # 2018-04-06 17:13:40.955356
    # [2018/04/06 17:13:40
    # [2018/04/06 17:13:40.955356
    (re.compile(r"^\[?(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})(?:\.(\d{6}))?"),
     lambda ts, ms: time.mktime(time.strptime(ts.replace("/", "-"), "%Y-%m-%d %H:%M:%S")) + float(ms or 0) / 1000000),

    # for strace logs
    # 16255 15:08:52.554223
    (re.compile(r"\d+ (\d{2}:\d{2}:\d{2}).(\d{6})"),
     lambda ts, ms: time.mktime(time.strptime("%s-%s-%s %s" % (YEAR, MONTH, DAY, ts), "%Y-%m-%d %H:%M:%S")) + float(ms) / 1000000),
]


class TimestampedStream(object):

    def __init__(self, stream, prefix="> "):
        self.name = stream.name if hasattr(stream, "name") else repr(stream)
        self.stream = iter(stream)
        self.prefix = prefix
        self.filler = " "*len(self.prefix)
        self._untimestamp = None

    def __gt__(self, other):
        return id(self) > id(other)

    def get_next(self):
        "Get next line in the stream and the stream itself, or None if stream ended"
        try:
            line = next(self.stream)
        except StopIteration:
            return
        else:
            ts = self.get_timestamp(uncolored(line))
            return ts, (self.prefix if ts else self.filler) + line, self

    def get_timestamp(self, line):
        """
        Find the timestamp if exists, and return as a float
        """
        if not line.startswith(" "):
            if not self._untimestamp:
                for regex, converter in TIMESTAMP_GETTERS:
                    match = regex.search(line)
                    if match:
                        # cache the regex and conversion funcs for later
                        self._untimestamp = converter, regex
                        break
            else:
                converter, regex = self._untimestamp
                match = regex.search(line)
            if match:
                return converter(*match.groups())
        return 0


def iter_zipped_logs(*log_streams, prefix="> ", show_intervals=None):
    """
    Line iterator that merges lines from different log streams based on their timestamp.
    Timestamp patterns are found in the TIMESTAMP_GETTERS list in this module.

    :param prefix: Prepend this prefix to each line where a timestamp was identified
    :param show_intervals: `s` or `ms` - Prepend the duration since the previous log line
    """

    # A sorted queue of (timestamp, stream) tuples (lowest-timestamp first)
    streams = PriorityQueue()
    stream_names = []
    for i, stream in enumerate(log_streams):
        if not isinstance(stream, tuple):
            tstream = TimestampedStream(stream, prefix)
        else:
            tstream = TimestampedStream(*stream)

        n = tstream.get_next()
        if n:
            stream_names.append(tstream.name)
            streams.put(n)

    last_ts = None
    if show_intervals:
        from easypy.units import Duration

        def formatted(line, current_ts, last_ts):
            fmt = "{:>7}{}"
            if (current_ts and last_ts):
                return fmt.format(Duration(current_ts - last_ts).render(show_intervals), line)
            else:
                return fmt.format("", line)
    else:
        def formatted(line, current_ts, last_ts):
            return line

    while not streams.empty():
        current_ts, line, stream = streams.get()
        yield formatted(line, current_ts, last_ts)
        last_ts = current_ts
        while True:
            n = stream.get_next()
            if not n:
                break   # stream ended
            ts, line, stream = n
            if ts and ts > current_ts:
                streams.put((ts, line, stream))
                break   # timestamp advanced
            yield formatted(line, ts, last_ts)
            if ts:
                last_ts = ts


def main():
    import sys
    import argparse
    parser = argparse.ArgumentParser(description='ZipLog - merge logs by timestamps')
    parser.add_argument(
        'logs', metavar='N', type=str, nargs='+',
        help='Log files; Use "-" for STDIN')
    parser.add_argument(
        '-i', '--interval', dest='interval', default=None,
        help="Show interval by seconds (s), or milliseconds (ms)")
    parser.add_argument(
        '-p', '--prefix', dest='prefix', default="> ",
        help="A prefix to prepend to timestamped lines")
    ns = parser.parse_args(sys.argv[1:])

    files = [sys.stdin if f == "-" else open(f) for f in ns.logs]
    try:
        for line in iter_zipped_logs(*files, show_intervals=ns.interval, prefix=ns.prefix):
            print(line, end="")
    except BrokenPipeError:
        pass


if __name__ == "__main__":
    main()
