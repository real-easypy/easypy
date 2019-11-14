def test_colors():
    from easypy.colors import Colorized, uncolored, colorize, register_colorizers

    register_colorizers(bad=("red", "blue"))

    opts = {
        str(Colorized("RED(BLUE)<<XXX>>")),
        str(Colorized("RED(BLUE)@[XXX]@")),
        str(Colorized("RED(BLUE)@{XXX}@")),

        str(colorize("RED(BLUE)<<XXX>>")),
        str(colorize("RED(BLUE)@[XXX]@")),
        str(colorize("RED(BLUE)@{XXX}@")),

        str(colorize("BAD<<XXX>>")),
        str(colorize("BAD@[XXX]@")),
        str(colorize("BAD@{XXX}@")),
    }

    assert len(opts) == 1
    [ret] = opts

    assert ret == "\x1b[1;44;31mXXX\x1b[0m"
    assert uncolored(ret) == "XXX"
