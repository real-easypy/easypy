from io import StringIO

from easypy.collections import defaultlist
from easypy.colors import colorize, uncolored
from easypy.humanize import compact


class Column():
    def __init__(self, name, title=None, max_width=None, align=None, header_align=None, padding=None, drop_if_empty=False):
        self.name = name
        self.max_width = max_width
        self.align = align
        self.header_align = header_align
        self.padding = padding
        self.overflow = 'ellipsis'
        self.title = title or name
        self.drop_if_empty = drop_if_empty
        self.visible = True


class Table():
    """
    :param List[Column] columns: column descriptors
    :param List[Bunch] data: rows
    """

    HEADER_SEP = "|"
    SEPARATORS = "|"
    BAR = '='
    BAR_SEP = ":"

    def __init__(self, *columns, data=None, max_col_width=None, align='left', header_align='center', padding=1):
        self.data = data or []
        self.columns = []

        self.max_col_width = max_col_width
        self.align = align
        self.header_align = header_align
        self.padding = padding

        for column in columns:
            self.add_column(column)

    _ALIGN_MAP = dict(left='<', right='>', center='^')

    def add_column(self, column: Column):
        self.columns.append(column)

    def add_row(self, **row):
        self.data.append(row)

    def render(self):
        rendered = defaultlist(list)
        columns = []

        def _get_value(data, value):
            ret = data.get(value)
            if ret is None:
                ret = ''
            return ret

        for column in self.columns:
            if not column.visible:
                continue
            rows = [_get_value(data, column.name) for data in self.data]
            if not any(filter(lambda i: i != '', rows)) and column.drop_if_empty:
                continue
            columns.append(column)

            if column.max_width is None:
                column.max_width = self.max_col_width
            if column.align is None:
                column.align = self.align
            if column.header_align is None:
                column.header_align = self.header_align
            if column.padding is None:
                column.padding = self.padding

            raw_data = [column.title] + rows
            colored_data = [colorize(str(data)) for data in raw_data]
            uncolored_data = [uncolored(data) for data in colored_data]
            max_width = column.max_width or max(len(data) for data in uncolored_data)
            for i, data in enumerate(colored_data):
                align = column.header_align if i == 0 else column.align
                coloring_spacing = len(colored_data[i]) - len(uncolored_data[i])
                spacing = max_width + coloring_spacing
                format_string = "{{data:{align}{spacing}}}".format(align=self._ALIGN_MAP[align], spacing=spacing)
                rendered[i].append(format_string.format(data=data))

        output = StringIO()
        for r_i, row in enumerate(rendered):
            r_parts = []

            sep = self.HEADER_SEP if r_i == 0 else self.SEPARATORS[r_i % len(self.SEPARATORS)]

            for col_i, col in enumerate(row):
                column = columns[col_i]
                padding = column.padding * " "
                if column.max_width and r_i > 0:
                    col = compact(col, column.max_width, suffix_length=column.max_width // 10)
                r_parts.append("{padding}{col}{padding}".format(col=col, padding=padding))

            output.write(sep.join(r_parts))
            output.write("\n")

            if r_i == 0:
                r_parts = [self.BAR * len(uncolored(part)) for part in r_parts]
                output.write(self.BAR_SEP.join(r_parts))
                output.write("\n")

        output.seek(0)
        return output.read()


class DecoratedTable(Table):
    HEADER_SEP = "│"
    SEPARATORS = "┼│┊┊│"
    BAR = '═'
    BAR_SEP = "╪"


def _test():
    table = Table(Column("first", "GREEN<<First>>"))
    table.add_column(Column("second", align='right'))
    table.add_row(first='1', second='BLUE<<longer>> second MAGENTA<<column>>')
    table.add_row(first='longer first column', second='2')
    print(table.render())
