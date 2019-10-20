from easypy.timing import Timer, TimeInterval


def test_time_interval1():

    st = 150000000
    t = Timer(st)
    t.t1 = t.t0 + 1

    ti = t.to_interval()

    assert t in ti

    assert t.t0 in ti
    assert t.t1 in ti

    assert t.t0 - 1 not in ti
    assert t.t1 + 1 not in ti

    assert ti.duration == t.duration
    assert ti.duration == 1
    assert ti.duration_delta.total_seconds() == 1

    assert str(ti) == '<TI 05:40:00..(1s)..05:40:01>'

    assert str(t.to_interval().to_timer()) == str(t)


def test_time_interval2():
    st = 150000000
    ti = TimeInterval()
    assert str(ti) == '<TI Eternity>'

    ti = TimeInterval(from_time=st)
    assert str(ti) == '<TI 05:40:00...>'

    ti = TimeInterval(from_time=st, to_time=st)
    assert str(ti) == '<TI 05:40:00..(0.0ms)..05:40:00>'

    ti = TimeInterval(to_time=st)
    assert str(ti) == '<TI ...05:40:00>'
