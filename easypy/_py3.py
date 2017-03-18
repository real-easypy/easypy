def raise_with_traceback(exception, tb):
    raise exception.with_traceback(tb) from None


from contextlib import ExitStack
from textwrap import indent
TimeoutError = TimeoutError
