import logging
from multiprocessing import Queue
from multiprocessing.shared_memory import SharedMemory

logger = logging.getLogger(__name__)


class SharedMemoryQueueResources:
    queue: Queue
    shared_memory_names: tuple[str, ...]

    def __init__(self, nbytes: int, size: int = 2):
        if size < 2:  # noqa: PLR2004
            msg = "size < 2"
            raise ValueError(msg)
        self.queue = Queue(size - 1)
        shared_memory = tuple(SharedMemory(create=True, size=nbytes) for _ in range(size))
        self.shared_memory_names = tuple(str(m.name) for m in shared_memory)
        for mem in shared_memory:
            logger.debug(f"Using shared memory '{mem.name}' for {self.queue}")
            mem.close()

    def __enter__(self) -> None:
        pass

    def __exit__(self, _type, value, traceback) -> None:
        self.close()

    def close(self) -> None:
        for name in self.shared_memory_names:
            mem = SharedMemory(name=name, create=False)
            mem.close()
            mem.unlink()


class SharedMemoryQueueManager:
    queues: list[Queue]

    def __init__(self):
        self.queues = []

    def make_queue(self, nbytes: int, size=2) -> SharedMemoryQueueResources:
        queue = SharedMemoryQueueResources(nbytes, size)
        self.queues.append(queue)
        return queue

    def __enter__(self) -> None:
        pass

    def __exit__(self, _type, value, traceback) -> None:
        for queue in self.queues:
            queue.close()
