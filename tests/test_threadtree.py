import threading

from easypy.threadtree import walk_frame_snapshots


def collect_frame_snapshots():
    """Collect only interesting frames"""
    result = []

    for frame in walk_frame_snapshots():
        if (
            "python" in str(frame)
            or "concurrency" in str(frame)
            or "/threadtree.py" in str(frame)
        ):
            continue
        result.append(frame)

    return result


def test_frame_snapshots():
    snapshots = None
    event_a = threading.Event()
    event_b= threading.Event()


    def func_b():
        nonlocal snapshots
        event_b.wait()
        snapshots = collect_frame_snapshots()


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
        "collect_frame_snapshots",
        "func_b",
        "func_a",
        "test_frame_snapshots",  # NOTE: must match the name of this test
        "<module>",
    ]
    actual = [frame.f_code_name for frame in snapshots]
    assert actual == expected
