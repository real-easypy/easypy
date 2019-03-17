from enum import Enum
from itertools import count
from datetime import datetime
from string import ascii_letters
from abc import ABCMeta, abstractmethod

from .tokens import AUTO
from .humanize import IndentableTextBuffer, ColumnsDiagram
from .collections import InfiniteIterator
from .properties import safe_property


class Timeline(object):
    class EventStyle(Enum):
        START = ' ┌'
        TICK = " ├"
        TWO_SIDED = "╶┼"
        TWO_SIDED_HEAVY = " ╺╈"
        END = ' └'
        STUB_END = ' ╵'

    class LineStyle(Enum):
        SOLID = ' │ '
        DASH = ' ╷ '
        DOT = ' ┇ '

    SYMBOLS = ascii_letters

    def __init__(self):
        self._index_generator = count(1)
        self._events = []

    def process(
            self,
            name: str,
            message: str,
            extra_info: str = '',
            *,
            event_style: EventStyle = EventStyle.START,
            line_style: LineStyle = LineStyle.SOLID,
            index: int = AUTO) -> 'TimelineProcess':
        if index is AUTO:
            index = next(self._index_generator)
        process = TimelineProcess(self, index, name, extra_info)
        process.event(message, event_style, line_style)

        return process

    def section(self, name: str, message: str = None):
        self._events.append(SectionEvent(
            timestamp=AUTO,
            name=name,
            message=message))

    def report(self, num_columns: int, style: dict = {}) -> IndentableTextBuffer:
        buf = IndentableTextBuffer()
        for row in TimelineEventsLogGenerator(self._events, num_columns, style):
            buf.write(row)
        return buf

    def symbol_for_index(self, index):
        return self.SYMBOLS[(index - 1) % len(self.SYMBOLS)]


class TimelineProcess(object):
    def __init__(self, timeline, index, name, extra_info):
        self._timeline = timeline
        self.index = index
        self.name = name
        self.extra_info = extra_info

        self._events = []

    def event(self, message: str, event_style: Timeline.EventStyle, line_style: Timeline.LineStyle = AUTO):
        if line_style is AUTO:
            line_style = self._events[-1].line_style
        event = TimelineProcessEvent(
            process=self,
            timestamp=AUTO,
            message=message,
            event_style=event_style,
            line_style=line_style)
        self._events.append(event)
        self._timeline._events.append(event)

    def finish(self, message: str, event_style: Timeline.EventStyle):
        self.event(
            message=message,
            event_style=event_style,
            line_style=None)

    @safe_property
    def symbol(self):
        return self._timeline.symbol_for_index(self.index)

    @safe_property
    def ident(self):
        return "%d:%s" % (self.index, self.symbol)


class TimelineEvent(metaclass=ABCMeta):
    def __init__(self, timestamp: datetime = AUTO):
        if timestamp is AUTO:
            timestamp = datetime.now()
        self.timestamp = timestamp

    @abstractmethod
    def gen_row_from_columns_diagram(self, columns_diagram: ColumnsDiagram):
        """Use the ``columns_diagram`` to generate the row of this event"""

    @safe_property
    @abstractmethod
    def ident(self):
        """A short identifier for the event"""

    @safe_property
    def short_timestamp(self):
        """A time-only timestamp of the event"""
        return self.timestamp.strftime('%H:%M:%S')

    @safe_property
    @abstractmethod
    def description(self):
        """The description of the event"""

    @safe_property
    @abstractmethod
    def process_extra_info(self):
        """Extra info of the process"""


class _EmptyEvent(TimelineEvent):
    def gen_row_from_columns_diagram(self, columns_diagram: ColumnsDiagram):
        return columns_diagram.empty()

    @safe_property
    def ident(self):
        return ''

    @safe_property
    def short_timestamp(self):
        return ''

    @safe_property
    def description(self):
        return ''

    @safe_property
    def process_extra_info(self):
        return ''
_EmptyEvent = _EmptyEvent()


class TimelineProcessEvent(TimelineEvent):
    def __init__(
            self,
            process: TimelineProcess,
            timestamp: datetime,
            message: str,
            event_style: Timeline.EventStyle,
            line_style: Timeline.LineStyle
    ):
        super().__init__(timestamp)
        self.process = process
        self.message = message
        self.event_style = event_style.value if isinstance(event_style, Enum) else event_style
        self.line_style = line_style.value if isinstance(line_style, Enum) else line_style

    def gen_row_from_columns_diagram(self, columns_diagram: ColumnsDiagram):
        if self.line_style is None:
            return columns_diagram.end(
                key=self.process.index,
                tick_style=self.event_style + self.symbol)
        else:
            return columns_diagram.change(
                key=self.process.index,
                tick_style=self.event_style + self.symbol,
                line_style=self.line_style)

    @safe_property
    def symbol(self):
        return self.process.symbol

    @safe_property
    def ident(self):
        return self.process.ident

    @safe_property
    def description(self):
        return '%s - %s' % (self.process.name, self.message)

    @safe_property
    def process_extra_info(self):
        return self.process.extra_info


class SectionEvent(TimelineEvent):
    def __init__(self, timestamp: datetime, name: str, message: str = None):
        super().__init__(timestamp)
        self.name = name
        self.message = message

    def gen_row_from_columns_diagram(self, columns_diagram: ColumnsDiagram):
        return columns_diagram.separator(self.name)

    @safe_property
    def ident(self):
        return ''

    @safe_property
    def description(self):
        if self.message:
            return '%s - %s' % (self.name, self.message)
        else:
            return self.name

    @safe_property
    def process_extra_info(self):
        return ''


class TimelineEventsLogGenerator(object):
    FORMAT_DLG = '|'.join([
        '{event.ident:>{style[ident_width]}}',
        ' {graph}  |',
        '{event.ident:>{style[ident_width]}}',
        ' {event.short_timestamp:9} ',
        ' {event.description:{style[description_width]}} ',
        ' {event.process_extra_info}',
    ]).format

    EMPTY_COLUMN = ' ' * 3

    def __init__(self, events, num_columns, style: dict = {}):
        self._events = events
        self._next_event_index = 0
        self._columns_diagram = ColumnsDiagram(num_columns)
        self._iter = InfiniteIterator(self._process_entry)
        self._style = dict(
            ident_width=6,
            description_width=65)
        self._style.update(style)

    def __iter__(self):
        return self._iter

    def _process_entry(self):
        if len(self._events) <= self._next_event_index:
            return

        event = self._events[self._next_event_index]
        self._next_event_index += 1

        graph = event.gen_row_from_columns_diagram(self._columns_diagram)

        yield self.FORMAT_DLG(event=event, graph=graph, style=self._style)
        yield self.FORMAT_DLG(event=_EmptyEvent, graph=self._columns_diagram.empty(), style=self._style)
