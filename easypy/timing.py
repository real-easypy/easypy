from contextlib import contextmanager
import time
from datetime import datetime, timedelta

from .decorations import parametrizeable_decorator
from .units import Duration


inf = float("inf")  # same as math.inf, but compatible with 3.4


def to_timestamp(t):
    return t.timestamp() if isinstance(t, datetime) else t


class TimeInterval(object):
    __slots__ = ("t0", "t1")

    NO_START = -inf
    NO_END = inf

    def __init__(self, from_time=NO_START, to_time=NO_END):
        self.t0 = self.NO_START if from_time is None else to_timestamp(from_time)
        self.t1 = self.NO_END if to_time is None else to_timestamp(to_time)

    @property
    def start_time(self):
        if self.t0 > self.NO_START:
            return datetime.fromtimestamp(self.t0)

    @property
    def end_time(self):
        if self.t1 < self.NO_END:
            return datetime.fromtimestamp(self.t1)

    @property
    def duration_delta(self):
        return timedelta(seconds=self.duration) if self.duration < inf else None

    @property
    def duration(self):
        return Duration(self.t1 - self.t0)

    def __contains__(self, t):
        """
        check if timestamp is within the timer's interval
        """

        if isinstance(t, Timer):
            t = t.to_interval()

        if isinstance(t, TimeInterval):
            return self.t0 <= t.t0 and t.t1 <= self.t1

        t = to_timestamp(t)
        return self.t0 <= t <= self.t1

    def to_timer(self):
        assert self.t0 > self.NO_START, "Can't convert a start-less TimeInterval to a Timer"
        t = Timer(self.t0)
        t.t1 = self.t1 if self.t1 < self.NO_END else None
        return t

    def render(self):
        t0, duration, t1 = self.t0, self.duration, self.t1
        if (t0, t1) == (self.NO_START, self.NO_END):
            return "Eternity"

        parts = [".", "."]

        if t0 > self.NO_START:
            parts[0] = time.strftime("%T", time.localtime(t0))

        if t1 < self.NO_END:
            parts[-1] = time.strftime("%T", time.localtime(self.t1))

        if duration < inf:
            parts.insert(1, "({})".format(duration))

        return "..".join(parts)

    def __str__(self):
        return "<TI %s>" % self.render()

    def __repr__(self):
        return "TimeInterval(%s)" % self.render()

    def _repr_pretty_(self, p, cycle):
        # used by IPython
        from easypy.colors import MAGENTA
        if cycle:
            p.text('TimeInterval(...)')
            return
        p.text(MAGENTA(self.render()))


class Timer(object):

    """
    Multi-purpose timer object::

        t = Timer()
        # do something ...
        t.stop()
        print(t.elapsed)

        t = Timer(expiration=120)
        while not t.expired:
            print("Time remaining: %r" % t.remain)
            # do something ...

        for remain in Timer(expiration=120).iter(sleep=2):
            print("Time remaining: %r" % remain)
            # do something ...
    """

    __slots__ = ("t0", "t1", "expiration")

    def __init__(self, now=None, expiration=None):
        self.reset(now)
        self.t1 = None
        self.expiration = None if expiration is None else Duration(expiration)

    def reset(self, now=None):
        self.t0 = now or time.time()

    def stop(self):
        self.t1 = time.time()
        return self.t1 - self.t0

    def iter(self, sleep=1):
        while not self.expired:
            time.sleep(sleep)
            yield self.remain

    def to_interval(self):
        return TimeInterval(self.t0, self.t1)

    def __contains__(self, timestamp):
        """
        check if timestamp is within the timer's interval
        """
        timestamp = to_timestamp(timestamp)
        if timestamp < self.t0:
            return False
        if not self.t1:
            return True
        if timestamp > self.t1:
            return False
        return True

    __iter__ = iter

    @property
    def stopped(self):
        return self.t1 is not None

    @property
    def duration_delta(self):
        return timedelta(seconds=self.duration)

    @property
    def duration(self):
        return Duration((self.t1 or time.time()) - self.t0)

    @property
    def elapsed_delta(self):
        return timedelta(seconds=time.time() - self.t0)

    @property
    def elapsed(self):
        return Duration(self.elapsed_delta.total_seconds())

    @property
    def expired(self):
        return Duration(max(0, self.elapsed - self.expiration) if self.expiration is not None else 0)

    @property
    def remain(self):
        return Duration(max(0, self.expiration - self.elapsed)) if self.expiration is not None else None

    @property
    def start_time(self):
        return datetime.fromtimestamp(self.t0)

    @property
    def stop_time(self):
        if self.t1 is None:
            return None
        else:
            return datetime.fromtimestamp(self.t1)

    def render(self):
        t0, duration, elapsed, expired, stopped = self.t0, self.duration, self.elapsed, self.expired, self.stopped
        st = time.strftime("%T", time.localtime(t0))
        if expired:
            duration = "{}+{}".format(self.expiration, expired)
        if stopped:
            et = time.strftime("%T", time.localtime(self.t1))
            fmt = "{st}..({duration})..{et}"
        elif expired:
            fmt = "{st}..({duration})"
        else:
            fmt = "{st}..({duration})"
        return fmt.format(**locals())

    def __str__(self):
        return "<T %s>" % self.render()

    def __repr__(self):
        return "Timer(%s)" % self.render()

    def _repr_pretty_(self, p, cycle):
        # used by IPython
        from easypy.colors import MAGENTA, RED
        if cycle:
            p.text('Timer(...)')
            return
        p.text((RED if self.expired else MAGENTA)(self.render()))


class BackoffTimer(Timer):

    def __init__(self, expiration, now=None, backoff_every=5, backoff_by=5, max_interval=None):
        super().__init__(expiration=expiration, now=now)
        self.backoff_by = backoff_by
        self.backoff_every = backoff_every
        self.iteration = 0
        self.max_interval = max_interval

    def backoff(self):
        self.iteration += 1
        if self.iteration % self.backoff_every == 0:
            self.expiration *= self.backoff_by
            if self.max_interval:
                self.expiration = min(self.max_interval, self.expiration)
        super().reset()


class StopWatch(object):
    def __init__(self):
        self._last_pause = 0
        self._timer = Timer()
        self.paused = True

    @property
    def elapsed(self):
        if self.paused:
            return self._last_pause
        return self._last_pause + self._timer.elapsed

    def pause(self):
        self._last_pause += self._timer.elapsed
        self.paused = True

    def start(self):
        self.paused = False
        self._timer.reset()

    def reset_and_start(self, last_paused=None):
        self._last_pause = 0
        self.start()


@contextmanager
def timing(t=None):
    if not t:
        t = Timer()
    try:
        yield t
    finally:
        t.stop()


def repeat(timeout, callback, sleep=0.5, progressbar=True):
    pred = lambda: callback() and False  # prevent 'wait' from stopping when the callback returns a nonzero
    return wait(timeout, pred=pred, sleep=sleep, progressbar=progressbar, throw=False)


@parametrizeable_decorator
def at_period(func, period=None):
    if not period:
        raise NotImplementedError('must specify a period')
    timer = Timer(expiration=period)

    def inner(*args, **kwargs):
        if timer.expired:
            ret = func(*args, **kwargs)
            timer.reset()
            return ret
    return inner


class StateTimeHistogram(object):
    def __init__(self):
        self._state = None
        self._states_times = {}
        self._timer = Timer()

    def _update_state_time(self, states_times=None):
        if states_times is None:
            states_times = self._states_times
        if self._state is not None:
            states_times[self._state] = states_times.get(self._state, 0) + self._timer.elapsed

    def finish(self):
        self._update_state_time()
        self._state = None
        self._timer.stop()

    def set_state(self, state):
        self._update_state_time()
        self._state = state
        self._timer.reset()

    @property
    def states_times(self):
        states_times = dict(self._states_times)
        self._update_state_time(states_times)
        return states_times


def throttled(duration):
    """
    Syntax sugar over timecache decorator.

    With accent on throttling calls and not actual caching of values Concurrent
    callers will block if function is executing, since they might depend on
    side effect of function call.
    """
    from easypy.caching import timecache
    return timecache(expiration=duration)


# re-exports
from .sync import wait, PredicateNotSatisfied, TimeoutException, iter_wait, wait_progress, iter_wait_progress
