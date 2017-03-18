import datetime
import time
import re

ISO_TIMEZONE_PATTERN = re.compile(r'^([+-])(\d{2}):?(\d{2})$')


class TimeZone(datetime.tzinfo):
    def __init__(self, offset, tzname=None, dst_hours_dlg=None):
        if isinstance(offset, datetime.timedelta):
            self.offset = offset
        else:
            self.offset = datetime.timedelta(hours=offset)

        self.dst_hours_dlg = dst_hours_dlg

        self.offset_total_seconds = self.offset.days * 24 * 3600 + self.offset.seconds
        if tzname:
            self._tzname = tzname
        else:
            if 0 == self.offset_total_seconds:
                self._tzname = 'UTC'
            elif 0 < self.offset_total_seconds:
                self._tzname = 'UTC+%02d:%02d' % (self.offset_total_seconds / 3600, self.offset_total_seconds % 3600 / 60)
            else:
                self._tzname = 'UTC-%02d:%02d' % (-self.offset_total_seconds / 3600, -self.offset_total_seconds % 3600 / 60)

    @classmethod
    def parse(cls, source):
        if source == 'Z':
            return cls.utc
        m = ISO_TIMEZONE_PATTERN.match(source)
        if m:
            offset = (int(m.group(2)) * 60 + int(m.group(3))) * 60
            if '-' == m.group(1):
                return cls(datetime.timedelta(seconds=-offset))
            else:
                return cls(datetime.timedelta(seconds=offset))
        else:
            return None

    def utcoffset(self, dt):
        dst = self.dst(dt)
        if dst:
            return self.offset + dst
        else:
            return self.offset

    def dst(self, dt):
        if self.dst_hours_dlg:
            return datetime.timedelta(hours=self.dst_hours_dlg(dt))
        else:
            return None

    def fromutc(self, dt):
        dt += dt.utcoffset()
        return dt

    def tzname(self, dt):
        return self._tzname

    def now(self):
        return datetime.datetime.now(self)

    @property
    def min(self):
        return datetime.datetime.min.replace(tzinfo=self)

    @property
    def max(self):
        return datetime.datetime.max.replace(tzinfo=self)

    def __repr__(self):
        return "<%s %r>" % (self.__class__.__name__, self._tzname)


TimeZone.utc = TimeZone(0, 'UTC')
TimeZone.local = TimeZone(datetime.timedelta(seconds=-time.timezone), time.tzname[0], dst_hours_dlg=(lambda dt: time.localtime().tm_isdst) if time.daylight else None)

utc_now = TimeZone.utc.now

UTC_TIMEZONE_PATTERN = re.compile(r'^(?P<Y>\d{4})-(?P<m>\d{2})-(?P<d>\d{2})T(?P<H>\d{2}):(?P<M>\d{2}):(?P<S>\d{2})(?P<MS>\.\d{1,6})?\d*(?P<tz>Z|[+-]\d{2}:?\d{2})?$')

ISO_PATTERN_WITH_MS = '%Y-%m-%dT%H:%M:%S.%f'
ISO_PATTERN_WITHOUT_MS = '%Y-%m-%dT%H:%M:%S'


def parse_isoformat(source):
    m = UTC_TIMEZONE_PATTERN.match(source)
    if m:
        timestamp = datetime.datetime(*(int(m.group(part)) for part in ['Y', 'm', 'd', 'H' ,'M' ,'S']))
        if m.group('MS'):
            ms_txt = m.group('MS')[1 :] # remove the dot
            ms = int(ms_txt) * 10 ** (6 - len(ms_txt))
            timestamp = timestamp.replace(microsecond=ms)
        if m.group('tz'):
            timestamp = timestamp.replace(tzinfo = TimeZone.parse(m.group('tz')))
        return timestamp


def test_conversions():
    for timestamp in [
            '2001-02-03T04:05:06',
            '2001-02-03T04:05:06.000007',
            '2001-02-03T04:05:06+08:00',
            '2001-02-03T04:05:06.000007+08:00',
            ]:
        assert parse_isoformat(timestamp).isoformat() == timestamp
    assert parse_isoformat('2001-02-03T04:05:06.000007Z') == datetime.datetime(2001, 2, 3, 4, 5, 6, 7, TimeZone.utc)
    assert parse_isoformat('2001-02-03T04:05:06+02:00').astimezone(TimeZone.utc) == parse_isoformat('2001-02-03T00:05:06-02:00').astimezone(TimeZone.utc)
    assert parse_isoformat('2001-02-03T04:05:06+02:00').utcoffset().seconds == 2 * 3600


short_utc_fmt = lambda dt: "%s,%02dZ" % (dt.strftime("%T"), dt.microsecond//10000)

if __name__=="__main__":
    test_conversions()
utc_ts = lambda: int(TimeZone.utc.now().timestamp())