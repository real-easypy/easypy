def test_thread_stacks():
    from easypy.threadtree import get_thread_stacks
    from easypy.concurrency import concurrent
    from time import sleep

    with concurrent(sleep, .1, threadname='sleep'):
        print(get_thread_stacks().render())
