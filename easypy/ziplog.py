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

    # [2018/04/06 17:13:40.955356,
    (re.compile(r"\[(\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2})\.(\d{6}),"),
     lambda ts, ms: time.mktime(time.strptime(ts, "%Y/%m/%d %H:%M:%S")) + float(ms) / 1000000),

    # 2018-04-06 17:13:40,955
    # 2018-04-23 04:48:11,811|
    (re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d{3})[| ]"),
     lambda ts, ms: time.mktime(time.strptime(ts, "%Y-%m-%d %H:%M:%S")) + float(ms) / 1000),

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


def iter_zipped_logs(*log_streams, prefix="DARK_GRAY@{> }@"):
    "Line iterator that merges lines from different log streams based on their timestamp"

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

    while not streams.empty():
        current_ts, line, stream = streams.get()
        yield line
        while True:
            n = stream.get_next()
            if not n:
                break   # stream ended
            ts, line, stream = n
            if ts and ts > current_ts:
                streams.put((ts, line, stream))
                break   # timestamp advanced
            yield line


if __name__ == "__main__":
    import sys
    files = map(open, sys.argv[1:])
    try:
        for line in iter_zipped_logs(*files):
            print(line, end="")
    except BrokenPipeError:
        pass
