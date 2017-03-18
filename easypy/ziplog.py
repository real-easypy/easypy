import re
import time
from queue import PriorityQueue
from .colors import uncolorize
from .logging import PROGRESS_BAR


TIMESTAMP_PATTERN = "%Y-%m-%d %H:%M:%S"


def to_timestamp(t):
    return "-".center(19) if t is None else time.strftime(TIMESTAMP_PATTERN, time.localtime(t))


YEAR = time.strftime("%Y")
TIMESTAMP_GETTERS = (

    (re.compile("^(\w{3} +\d+ +\d+:\d+:\d+)"),
     lambda ts: time.mktime(time.strptime("%s %s" % (YEAR, ts), "%Y %b %d %H:%M:%S"))),
    (re.compile("(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d{3})\|"),
     lambda ts, ms: time.mktime(time.strptime(ts, "%Y-%m-%d %H:%M:%S")) + float(ms)/100000),

    )


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
            ts = self.get_timestamp(uncolorize(line))
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
        return None


def iter_zipped_logs(*log_streams):
    "Line iterator that merges lines from different log streams based on their timestamp"

    # A sorted queue of (timestamp, stream) tuples (lowest-timestamp first)
    streams = PriorityQueue()
    stream_names = []
    for i, stream in enumerate(log_streams):
        if not isinstance(stream, tuple):
            tstream = TimestampedStream(stream, "DARK_GRAY@{> }@")
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
    for line in iter_zipped_logs(*files):
        print(line, end="")
