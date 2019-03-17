from easypy.timeline import Timeline


def test_timeline():
    timeline = Timeline()

    p1 = timeline.process('my-process', 'custom start message')

    p1.event('rebuilding', timeline.EventStyle.TWO_SIDED, timeline.LineStyle.DASH)

    p2 = timeline.process('my-other-process', 'blank message', 'some params')
    p2.event('Le thingie', timeline.EventStyle.TICK)

    timeline.section('hi', 'stage start')

    p2.event('Le thingie', timeline.EventStyle.TICK)
    p1.event('Something', timeline.EventStyle.TICK)

    p1.finish('abort', timeline.EventStyle.END)

    print()
    print('==TIMELINE REPORT==')
    report = timeline.report(num_columns=2, style=dict(ident_width=3))
    print('-------------------')
    print(report)
    print('--TIMELINE REPORT--')
