import datetime
from easypy.timezone import TimeZone, parse_isoformat


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
