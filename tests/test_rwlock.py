import threading
import pytest
import logging
from time import sleep
from easypy.concurrency import concurrent
from easypy.sync import RWLock, TimeoutException
from easypy.bunch import Bunch
from easypy.sync import RWLock


def test_rwlock():

    main_ctrl = threading.Event()
    reader_ctrl = threading.Event()
    writer_ctrl = threading.Event()
    lock = RWLock("test")

    state = Bunch(reading=False, writing=False)

    def read():
        logging.info("Before read")
        reader_ctrl.wait()
        reader_ctrl.clear()

        with lock:
            logging.info("Reading...")
            state.reading = True
            main_ctrl.set()
            reader_ctrl.wait()
            reader_ctrl.clear()
            state.reading = False
            logging.info("Done reading")

        logging.info("After read")

    def write():
        logging.info("Before write")
        writer_ctrl.wait()
        writer_ctrl.clear()

        with lock.exclusive():
            logging.info("Writing...")
            state.writing = True
            main_ctrl.set()
            writer_ctrl.wait()
            writer_ctrl.clear()
            state.writing = False
            logging.info("Done writing")

        logging.info("After write")
        main_ctrl.set()

    reader = concurrent(read, threadname='read')
    writer = concurrent(write, threadname='write')

    with reader, writer:
        assert not state.reading and not state.writing

        reader_ctrl.set()
        main_ctrl.wait()
        logging.info("lock acquired non-exclusively")
        main_ctrl.clear()
        assert state.reading and not state.writing

        writer_ctrl.set()
        logging.info("writer awaits exclusivity")
        with lock:
            assert state.reading and not state.writing

        reader_ctrl.set()
        main_ctrl.wait()
        main_ctrl.clear()
        logging.info("read lock released")
        assert not state.reading and state.writing

        writer_ctrl.set()
        main_ctrl.wait()
        main_ctrl.clear()
        logging.info("write lock released")
        assert not state.reading and not state.writing


def test_rwlock_different_threads():
    lock = RWLock("test")
    ea = threading.Event()
    eb = threading.Event()

    def a(id=None):
        lock.acquire(id)
        ea.set()
        eb.wait()

    def b(id=None):
        lock.release(id)
        eb.set()

    with concurrent(a):
        ea.wait()
        assert lock.owners
        with pytest.raises(RuntimeError):
            b()
        eb.set()
        assert lock.owners

    with concurrent(a, "same"):
        assert lock.owners
        with concurrent(b, "same"):
            pass
        assert lock.owners


def test_wrlock_exclusive_timeout():
    wrlock = RWLock()
    
    def acquire_lock():
        nonlocal wrlock
        wrlock.acquire()
        sleep(1)
        wrlock.release()
    
    t1 = threading.Thread(target=acquire_lock)
    t1.start()
    
    sleep(0.01)
    with pytest.raises(TimeoutException):
        with wrlock.exclusive(timeout=0.5):
            pass
    
    t1.join()
