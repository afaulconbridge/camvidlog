import logging
from collections.abc import Iterable
from contextlib import contextmanager
from multiprocessing import JoinableQueue
from multiprocessing.shared_memory import SharedMemory
from typing import Any

import numpy as np

from camvidlog.frameinfo import FrameQueueInfoOutput

logger = logging.getLogger(__name__)


class SharedMemoryFrameContext:
    def __init__(self, frame_queue_info: FrameQueueInfoOutput, shared_memory_name: str, *extras: Iterable[Any]):
        self.shared_memory_name = shared_memory_name
        self.frame_queue_info = frame_queue_info
        self.extras = extras

    def __enter__(self):
        self.shared_memory = SharedMemory(name=self.shared_memory_name, create=False)
        self.array = np.ndarray(
            self.frame_queue_info.shape,
            dtype=np.uint8,
            buffer=self.shared_memory.buf,
        )
        return (self.array, *self.extras)

    def __exit__(self, _type, value, traceback) -> None:
        self.shared_memory.close()
        self.frame_queue_info.queue.task_done()


class SharedMemoryQueueConsumer:
    frame_queue_info: FrameQueueInfoOutput

    def __init__(self, frame_queue_info: FrameQueueInfoOutput):
        self.frame_queue_info = frame_queue_info

    def get(self, block: bool = True, timeout: float | None = None) -> Any:  # noqa: FBT001, FBT002
        item = self.frame_queue_info.queue.get(block, timeout)
        if item is None:
            self.frame_queue_info.queue.task_done()
            return None
        else:
            shared_memory_name, *extras = item
            return SharedMemoryFrameContext(self.frame_queue_info, shared_memory_name, *extras)


class SharedMemoryQueueResources:
    queue: JoinableQueue
    shared_memory_names: tuple[str, ...]

    def __init__(self, nbytes: int, size: int = 3):
        if size < 2:  # noqa: PLR2004
            msg = "size < 2"
            raise ValueError(msg)
        self.queue = JoinableQueue(size - 2)
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
        self.queue.join()
        for name in self.shared_memory_names:
            mem = SharedMemory(name=name, create=False)
            mem.close()
            mem.unlink()


class SharedMemoryQueueManager:
    queues: list[SharedMemoryQueueResources]

    def __init__(self):
        self.queues = []

    def make_queue(self, nbytes: int, size=3) -> SharedMemoryQueueResources:
        queue = SharedMemoryQueueResources(nbytes, size)
        self.queues.append(queue)
        return queue

    def __enter__(self) -> None:
        pass

    def __exit__(self, _type, value, traceback) -> None:
        for queue in self.queues:
            queue.close()
