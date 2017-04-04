from easypy.threadtree import get_thread_stacks, ThreadContexts
from easypy.concurrency import concurrent
from time import sleep


def test_thread_stacks():
    with concurrent(sleep, .1, threadname='sleep'):
        print(get_thread_stacks().render())


def test_thread_contexts_counters():
    TC = ThreadContexts(counters=('i', 'j'))
    assert TC.i == TC.j == 0

    with TC(i=1):
        def check1():
            assert TC.i == 1
            assert TC.j == 0

            with TC(i=1, j=1):
                def check2():
                    assert TC.i == 2
                    assert TC.j == 1
                with concurrent(check2):
                    pass

        with concurrent(check1):
            pass


def test_thread_context_stacks():
    TC = ThreadContexts(stacks=('i', 'j'))
    assert TC.i == TC.j == []

    with TC(i='a'):
        def check1():
            assert TC.i == ['a']
            assert TC.j == []

            with TC(i='i', j='j'):
                def check2():
                    assert TC.i == ['a', 'i']
                    assert TC.j == ['j']
                with concurrent(check2):
                    pass

        with concurrent(check1):
            pass
