import pytest
from easypy.colors import Colorized, uncolored, colorize, register_colorizers
register_colorizers(bad=("red", "blue"))


@pytest.mark.parametrize("content", ["XXX", ""])
def test_colors(content):
    opts = {
        str(Colorized("RED(BLUE)<<%s>>" % content)),
        str(Colorized("RED(BLUE)@[%s]@" % content)),
        str(Colorized("RED(BLUE)@{%s}@" % content)),

        str(colorize("RED(BLUE)<<%s>>" % content)),
        str(colorize("RED(BLUE)@[%s]@" % content)),
        str(colorize("RED(BLUE)@{%s}@" % content)),

        str(colorize("BAD<<%s>>" % content)),
        str(colorize("BAD@[%s]@" % content)),
        str(colorize("BAD@{%s}@" % content)),
    }

    assert len(opts) == 1
    [ret] = opts
    assert ret == ("\x1b[1;44;31m%s\x1b[0m" % content if content else '')
    assert uncolored(ret) == content


@pytest.mark.parametrize("content", ["XXX", ""])
def test_uncolored(content):
    uncolored(str(Colorized("RED(BLUE)<<%s>>" % content)))
    uncolored(str(Colorized("RED(BLUE)@[%s]@" % content)))
    uncolored(str(Colorized("RED(BLUE)@{%s}@" % content)))

    uncolored(str(colorize("RED(BLUE)<<%s>>" % content)))
    uncolored(str(colorize("RED(BLUE)@[%s]@" % content)))
    uncolored(str(colorize("RED(BLUE)@{%s}@" % content)))

    uncolored(str(colorize("BAD<<%s>>" % content)))
    uncolored(str(colorize("BAD@[%s]@" % content)))
    uncolored(str(colorize("BAD@{%s}@" % content)))

    uncolored("RED(BLUE)<<%s>>" % content)
    uncolored("RED(BLUE)@[%s]@" % content)
    uncolored("RED(BLUE)@{%s}@" % content)

    uncolored("RED(BLUE)<<%s>>" % content)
    uncolored("RED(BLUE)@[%s]@" % content)
    uncolored("RED(BLUE)@{%s}@" % content)

    uncolored("BAD<<%s>>" % content)
    uncolored("BAD@[%s]@" % content)
    uncolored("BAD@{%s}@" % content)
