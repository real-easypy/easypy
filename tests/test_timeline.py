import time

from easypy.timeline import Timeline, TimelineEntry
from easypy.properties import safe_property

def test_timeline():
    class ActionStart(TimelineEntry):
        def __init__(self, idx, name, timestamp):
            self.idx = idx
            self.name = name
            self.timestamp = timestamp

        @safe_property
        def idx_for_symbol(self):
            return self.idx

        def gen_log_entries(self, info_generator):
            yield info_generator.gen_entry(
                is_active=True,

                column_override={self.name: "╶%s " % self.symbol},
                ident="%s" % (self.symbol),
                timestamp=time.strftime('%H:%M:%S', time.localtime(self.timestamp)),
                description='%s - start' % (self.name))

    base_time = 1563470556

    timeline = Timeline()

    timeline.add_entry(ActionStart(1, 'foo', base_time + 1))

    print('==')
    print(timeline.report())
    print('==')
