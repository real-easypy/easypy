import pytest
from contextlib import contextmanager
from io import StringIO
from logging import getLogger, Formatter, StreamHandler
from easypy.colors import uncolored
logger = getLogger(__name__)


@pytest.yield_fixture
def get_log():
    level = logger.root.level
    stream = StringIO()
    handler = StreamHandler(stream)
    handler.setFormatter(Formatter(fmt="%(message)s"))
    logger.root.addHandler(handler)
    logger.root.setLevel(0)

    def get():
        return uncolored(stream.getvalue())
    yield get

    logger.root.setLevel(level)
    logger.root.removeHandler(handler)


def test_indent_around_generator(get_log):

    @logger.indented("hey")
    def gen():
        logger.info("000")
        yield 1
        yield 2

    for i in gen():
        logger.info("%03d", i)
        break

    assert get_log() == "hey\n000\n001\nDONE in no-time (hey)\n"


def test_indent_around_function(get_log):

    @logger.indented("hey")
    def foo():
        logger.info("001")

    foo()

    assert get_log() == "hey\n001\nDONE in no-time (hey)\n"


def test_indent_around_ctx(get_log):

    @logger.indented("hey")
    @contextmanager
    def ctx():
        logger.info("001")
        yield
        logger.info("003")

    with ctx():
        logger.info("002")

    assert get_log() == "hey\n001\n002\n003\nDONE in no-time (hey)\n"
