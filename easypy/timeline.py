from string import ascii_letters
import time
from weakref import WeakKeyDictionary
from abc import ABC, abstractmethod, abstractproperty

from easypy.bunch import Bunch
from easypy.humanize import time_duration, IndentableTextBuffer
from easypy.properties import safe_property
from easypy.tokens import AUTO

from logging import getLogger

_logger = getLogger(__name__)


class Timeline(object):
    def __init__(self, num_columns=8, logger=None):
        self.num_columns = num_columns
        self._entries = []
        self._generator_for_timeline_log = LogLinesInfoGenerator(self, num_columns=num_columns)
        self.logger = logger

    def add_entry(self, entry):
        idx_for_symbol = entry.idx_for_symbol
        if idx_for_symbol is not None:
            entry.symbol = self._symbol(idx_for_symbol)
        self._entries.append(entry)
        if self.logger:
            for entry in self._generator_for_timeline_log:
                self.logger.info(self.FORMAT.format(**entry))

    SYMBOLS = dict(enumerate(ascii_letters))
    FORMAT = '{ident:6}| {graph}  ||{ident:6}| {timestamp:9} | {description:65}'

    def _symbol(self, idx):
        return self.SYMBOLS[(idx - 1) % len(self.SYMBOLS)]

    def report(self, depth=0, num_columns=AUTO):
        if num_columns is AUTO:
            num_columns = self.num_columns
        history_length = len(self._entries)
        if depth:
            header = "Recent history (%s)" % (min(depth, history_length))
        else:
            header = "Complete history"
            depth = history_length

        rows_data = [
            params
            for params in LogLinesInfoGenerator(self, num_columns=num_columns)
            if (params.entry_number >= history_length - depth) or params.is_active]

        buff = IndentableTextBuffer(header)

        if rows_data:
            graph_width = max(len(row.graph) for row in rows_data)
            for row in rows_data:
                row.graph = row.graph.ljust(graph_width)
                buff.write(self.FORMAT, **row)

        return buff.render(edges=False, overflow="ignore")


class InfiniteIterator(object):
    """
    Iterator that calls the delegate and iterates on the values it yields

    * Once the sub-iteration finishes, it'll invoke the delegate again to see
      if there are new values.
    * If the iteration on the ``InfiniteIterator`` finishes, iterating on it
      again will invoke the delegate again and may yield new values.
    """

    def __init__(self, dlg):
        self._dlg = dlg
        self._current = iter(self._dlg())

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self._current)
        except StopIteration:
            pass
        self._current = iter(self._dlg())
        return next(self._current)


class LogLinesInfoGenerator(object):
    """
    Generator of timeline rows

    Rather than the row itself, returns a dictionary with info for creating the
    row (so that it can be used with `IndentableTextBuffer.write``)
    """

    SPACER = "  "
    WIDTH = 3
    EMPTY = " " * WIDTH

    def __init__(self, timeline, num_columns):
        self.timeline = timeline
        self.entry_number = 0
        assert 0 < num_columns
        self.columns = [self.EMPTY] * num_columns
        self.action_columns = WeakKeyDictionary()
        self._iter = InfiniteIterator(self._process_entry)

    def _extend_columns_array(self, num_columns):
        num_columns_needed = num_columns - len(self.columns)
        assert 0 < num_columns_needed
        self.columns += [self.EMPTY] * num_columns_needed

    def _assign_column(self, action):
        try:
            return self.action_columns[action]
        except KeyError:
            pass

        for col, ident in enumerate(self.columns):
            if ident == self.EMPTY:
                break
        else:  # no free columns )-;
            col = len(self.columns)
            self._extend_columns_array(len(self.columns) * 2)

        self.action_columns[action] = col
        return col

    def set_column(self, action, symbol):
        self.columns[self._assign_column(action)] = symbol

    def clear_column(self, action):
        try:
            column = self.action_columns[action]
        except KeyError:
            pass
        else:
            self.columns[column] = self.EMPTY

    def __iter__(self):
        return self._iter

    def _process_entry(self):
        if len(self.timeline._entries) <= self.entry_number:
            return

        entry_number = self.entry_number
        self.entry_number += 1

        entry = self.timeline._entries[entry_number]
        entry.perform_modifications(self)
        for log_entry in entry.gen_log_entries(self):
            log_entry.entry_number = entry_number
            yield log_entry

    def _gen_baes_entry(self):
        return Bunch(
            is_active=False,
            ident='',
            timestamp='',
            description='')

    def gen_entry(self, *, column_override={}, **params):
        column_override_by_index = {
            self.action_columns.get(action): override
            for action, override in column_override.items()}

        base_entry = self._gen_baes_entry()
        base_entry.graph = self.SPACER.join(
            column_override_by_index.get(i, col)
            for i, col in enumerate(self.columns))

        base_entry.update(params)

        return base_entry

    def graph_subcaption(self, text):
        target_width = (self.WIDTH * len(self.columns)) + (len(self.SPACER) * (len(self.columns) - 1))
        num_fill = target_width - len(text)
        if num_fill < 4:
            return '--' + text[:target_width - 4] + '--'
        fill_before = num_fill // 2
        fill_after = num_fill - fill_before
        return ('-' * fill_before) + text + ('-' * fill_after)


class TimelineEntry(ABC):
    def perform_modifications(self, info_generator):
        pass

    @safe_property
    def idx_for_symbol(self):
        return None

    @abstractmethod
    def gen_log_entries(self, info_generator):
        pass
