import threading

import logging
from easypy.concurrency import concurrent
from easypy.sync import RWLock
from easypy.bunch import Bunch


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
