from datetime import datetime, timedelta
from easypy.timezone import TimeZone, parse_isoformat, short_utc_fmt


def test_conversions():
    for timestamp in [
            '2001-02-03T04:05:06',
            '2001-02-03T04:05:06.000007',
            '2001-02-03T04:05:06+08:00',
            '2001-02-03T04:05:06.000007+08:00',
            ]:
        assert parse_isoformat(timestamp).isoformat() == timestamp
    assert parse_isoformat('2001-02-03T04:05:06.000007Z') == datetime(2001, 2, 3, 4, 5, 6, 7, TimeZone.utc)
    assert parse_isoformat('2001-02-03T04:05:06+02:00').astimezone(TimeZone.utc) == parse_isoformat('2001-02-03T00:05:06-02:00').astimezone(TimeZone.utc)
    assert parse_isoformat('2001-02-03T04:05:06+02:00').utcoffset().seconds == 2 * 3600


def test_short_utc_fmt():
    dt = datetime(2001, 2, 3, 4, 5, 6, 70000, TimeZone.utc)
    assert short_utc_fmt(dt) == '04:05:06,07Z'

    # No timezone
    assert short_utc_fmt(dt.replace(tzinfo=None)) == '04:05:06,07'

    # Nameless timezones - use the offset to display
    nameless_timezone = TimeZone(offset=timedelta(hours=6))
    assert short_utc_fmt(dt.astimezone(nameless_timezone)) == '10:05:06,07+06:00'

    nameless_timezone_2 = TimeZone(offset=timedelta(hours=-3))
    assert short_utc_fmt(dt.astimezone(nameless_timezone_2)) == '01:05:06,07-03:00'

    # Weka Time - a fake timezone that doesn't have DST
    wkt = TimeZone(offset=timedelta(hours=5), tzname='WKT')
    assert short_utc_fmt(dt.astimezone(wkt)) == '09:05:06,07WKT'
