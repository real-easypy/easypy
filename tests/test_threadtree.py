import threading

from easypy.threadtree import walk_frames


def collect_frames():
    """Collect only interesting frames"""
    result = []

    for frame in walk_frames(across_threads=True):
        if (
            "python" in str(frame)
            or "concurrency" in str(frame)
            or "/threadtree.py" in str(frame)
        ):
            continue
        result.append(frame)

    return result


def test_walk_frames():
    frames = []
    event_a = threading.Event()
    event_b = threading.Event()

    def func_b():
        nonlocal frames
        event_b.wait()
        frames = collect_frames()

    t2 = threading.Thread(target=func_b)

    def func_a():
        event_a.wait()
        t2.start()

    t1 = threading.Thread(target=func_a)
    t1.start()

    event_a.set()
    event_b.set()

    t1.join()
    t2.join()

    expected = [
        "collect_frames",
        "func_b",
        "func_a",
        "test_walk_frames",  # NOTE: must match the name of this test
        "<module>",
    ]
    actual = [frame.f_code.co_name for frame in frames]
    assert actual == expected
