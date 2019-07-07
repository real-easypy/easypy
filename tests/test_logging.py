import pytest
from contextlib import contextmanager
from io import StringIO
from easypy.colors import uncolored


@pytest.yield_fixture
def get_log(is_logbook):
    stream = StringIO()

    if is_logbook:
        import logbook
        handler = logbook.StreamHandler(stream, format_string="{record.message}")
        handler.push_application()
    else:
        import logging
        orig_level = logging.root.level
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter(fmt="%(message)s"))
        logging.root.addHandler(handler)
        logging.root.setLevel(0)

    def get():
        return uncolored(stream.getvalue())
    yield get

    if is_logbook:
        handler.pop_application()
    else:
        logging.root.setLevel(orig_level)
        logging.root.removeHandler(handler)


@pytest.fixture
def logger(request):
    from easypy.logging import _get_logger
    return _get_logger(name=request.function.__name__)


def test_indent_around_generator(get_log, logger):

    @logger.indented("hey")
    def gen():
        logger.info("000")
        yield 1
        yield 2

    for i in gen():
        logger.info("%03d" % i)
        break

    assert get_log() == "hey\n000\n001\nDONE in no-time (hey)\n"


def test_indent_around_function(get_log, logger):

    @logger.indented("hey")
    def foo():
        logger.info("001")

    foo()

    assert get_log() == "hey\n001\nDONE in no-time (hey)\n"


def test_indent_around_ctx(get_log, logger):

    @logger.indented("hey")
    @contextmanager
    def ctx():
        logger.info("001")
        yield
        logger.info("003")

    with ctx():
        logger.info("002")

    assert get_log() == "hey\n001\n002\n003\nDONE in no-time (hey)\n"
