import logging
from collections.abc import Iterable
from multiprocessing import JoinableQueue
from multiprocessing.shared_memory import SharedMemory
from types import TracebackType
from typing import Any, Self

import numpy as np

from camvidlog.frameinfo import FrameInfo

logger = logging.getLogger(__name__)


class SharedMemoryConsumerContext:
    frame_info: FrameInfo
    queue: JoinableQueue
    extras: tuple[Any]
    shared_memory: SharedMemory | None
    array: np.ndarray | None

    def __init__(self, frame_info: FrameInfo, queue: JoinableQueue, shared_memory_name: str, *extras: Iterable[Any]):
        self.frame_info = frame_info
        self.queue = queue
        self.shared_memory_name = shared_memory_name
        self.extras = extras
        self.shared_memory = None
        self.array = None

    def __enter__(self) -> tuple[np.ndarray, *tuple[Any, ...]]:
        self.shared_memory = SharedMemory(name=self.shared_memory_name, create=False)
        self.array = np.ndarray(
            self.frame_info.shape,
            dtype=np.uint8,
            buffer=self.shared_memory.buf,
        )
        return (self.array, *self.extras)

    def __exit__(
        self, type_: type[BaseException] | None, value: BaseException | None, traceback: TracebackType | None
    ) -> None:
        if self.shared_memory:
            self.shared_memory.close()
        self.queue.task_done()


class SharedMemoryConsumer:
    frame_info: FrameInfo
    queue: JoinableQueue
    timeout_default: float | None

    def __init__(self, frame_queue_info: FrameInfo, queue: JoinableQueue, timeout_default: float | None = None):
        self.frame_info = frame_queue_info
        self.queue = queue
        self.timeout_default = timeout_default

    def __iter__(self) -> Self:
        return self

    def __next__(self):
        next_ = self.get(timeout=self.timeout_default)
        if next_ is not None:
            return next_
        else:
            raise StopIteration

    def get(self, block: bool = True, timeout: float | None = None) -> SharedMemoryConsumerContext:  # noqa: FBT001, FBT002
        item = self.queue.get(block, timeout)
        if item is None:
            self.queue.task_done()
            return None
        else:
            shared_memory_name, *extras = item
            return SharedMemoryConsumerContext(self.frame_info, self.queue, shared_memory_name, *extras)


class SharedMemoryProducerContext:
    frame_info: FrameInfo
    queue: JoinableQueue
    extras: tuple[Any]
    shared_memory: SharedMemory | None
    array: np.ndarray | None
    block: bool
    timeout: float | None

    def __init__(
        self,
        frame_info: FrameInfo,
        queue: JoinableQueue,
        shared_memory_name: str,
        block: bool = True,  # noqa: FBT001, FBT002
        timeout: float | None = None,
    ):
        self.frame_info = frame_info
        self.queue = queue
        self.shared_memory_name = shared_memory_name
        self.extras = []
        self.shared_memory = None
        self.array = None
        self.block = block
        self.timeout = timeout

    def __enter__(self) -> tuple[np.ndarray, *tuple[Any, ...]]:
        self.shared_memory = SharedMemory(name=self.shared_memory_name, create=False)
        self.array = np.ndarray(
            self.frame_info.shape,
            dtype=np.uint8,
            buffer=self.shared_memory.buf,
        )
        return (self.array, self.extras)

    def __exit__(
        self, type_: type[BaseException] | None, value: BaseException | None, traceback: TracebackType | None
    ) -> None:
        item = (self.shared_memory_name, *self.extras)
        self.queue.put(item, self.block, self.timeout)
        if self.shared_memory is not None:
            self.shared_memory.close()


class SharedMemoryProducer:
    frame_info: FrameInfo
    queue: JoinableQueue
    timeout_default: float | None
    shared_memory_names: tuple[str, ...]
    shared_memory_name_index: int

    def __init__(self, frame_info: FrameInfo, queue: JoinableQueue, timeout_default: float | None = None):
        self.frame_info = frame_info
        self.queue = queue
        self.timeout_default = timeout_default
        shared_memory = tuple(SharedMemory(create=True, size=frame_info.nbytes) for _ in range(queue.maxsize + 2))
        self.shared_memory_names = tuple(str(m.name) for m in shared_memory)
        self.shared_memory_name_index = 0

    def __iter__(self) -> Self:
        return self

    def __next__(self):
        next_ = self.get(timeout=self.timeout_default)
        if next_ is not None:
            return next_
        else:
            raise StopIteration

    def get(self, block: bool = True, timeout: float | None = None) -> SharedMemoryProducerContext:  # noqa: FBT001, FBT002
        shared_memory_name = self.shared_memory_names[self.shared_memory_name_index]
        self.shared_memory_name_index -= 1
        if self.shared_memory_name_index <= 0:
            self.shared_memory_name_index = len(self.shared_memory_names) - 1

        return SharedMemoryProducerContext(self.frame_info, self.queue, shared_memory_name, block, timeout)

    def close(self) -> None:
        for name in self.shared_memory_names:
            mem = SharedMemory(name=name, create=False)
            mem.close()
            mem.unlink()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self, type_: type[BaseException] | None, value: BaseException | None, traceback: TracebackType | None
    ) -> None:
        self.close()


class SharedMemoryQueueResources:
    queue: JoinableQueue
    shared_memory_names: tuple[str, ...]

    def __init__(self, nbytes: int, size: int = 3):
        if size < 2:  # noqa: PLR2004
            msg = "size < 2"
            raise ValueError(msg)
        self.queue = JoinableQueue(size)
        shared_memory = tuple(SharedMemory(create=True, size=nbytes) for _ in range(size + 2))
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
