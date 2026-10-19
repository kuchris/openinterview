import threading

from openinterview.queues import DropOldestQueue


def test_drop_oldest_when_full():
    q = DropOldestQueue[int](3)
    assert [q.put(i) for i in range(3)] == [None, None, None]
    assert q.put(3) == 0
    assert q.put(4) == 1
    assert q.dropped == 2
    assert [q.get(0), q.get(0), q.get(0)] == [2, 3, 4]


def test_get_times_out_when_empty():
    q = DropOldestQueue[int](1)
    assert q.get(timeout=0.01) is None


def test_close_unblocks_consumer():
    q = DropOldestQueue[int](1)
    out = []
    t = threading.Thread(target=lambda: out.append(q.get(timeout=5)))
    t.start()
    q.close()
    t.join(1)
    assert out == [None]


def test_put_never_blocks_producer():
    q = DropOldestQueue[int](2)
    for i in range(10_000):
        q.put(i)
    assert len(q) == 2 and q.dropped == 9_998
