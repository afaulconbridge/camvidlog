from dataclasses import dataclass
from enum import Enum
from multiprocessing import Queue
from multiprocessing.shared_memory import SharedMemory
from typing import Generator

import cv2
import numpy as np

TIMEOUT = 60.0


class Resolution(Enum):
    # Y,X to match openCV
    VGA = (480, 640)
    SD = (720, 1280)
    HD = (1080, 1920)  # 2k
    UHD = (2160, 3840)  # 4K


@dataclass
class VideoFileStats:
    filename: str
    fps: float
    x: int
    y: int
    bw: bool
    nbytes: int
    dtype: np.dtype

    @property
    def shape(self):
        # Y,X to match openCV
        return (self.y, self.x, 1 if self.bw else 3)


def peek_in_file(filename: str) -> VideoFileStats:
    video_capture = None
    try:
        video_capture = cv2.VideoCapture(filename, cv2.CAP_ANY)
        fps = float(video_capture.get(cv2.CAP_PROP_FPS))
        # frame details are more reliable than capture properties
        # x = cv2.CAP_PROP_FRAME_WIDTH
        # y = cv2.CAP_PROP_FRAME_HEIGHT
        # bw = cv2.CAP_PROP_MONOCHROME
        (success, frame) = video_capture.read()
        if not success:
            msg = f"Unable to read frame from {filename}"
            raise RuntimeError(msg)
        y = frame.shape[0]
        x = frame.shape[1]
        bw = frame.shape[2] == 1
        return VideoFileStats(filename=filename, fps=fps, x=x, y=y, bw=bw, nbytes=frame.nbytes, dtype=frame.dtype)
    finally:
        if video_capture:
            video_capture.release()
            video_capture = None


class FileReader:
    videopath: str
    shared_memory_names: tuple[str, ...]
    shared_memory: tuple[SharedMemory, ...]
    shared_arrays: tuple[np.ndarray]
    shared_pointer: int
    fps: float
    queue: Queue
    video_capture: cv2.VideoCapture | None

    def __init__(
        self,
        videopath: str,
        fps: float,
        queue: Queue,
        shared_memory_names: tuple[str, ...],
        shape: tuple[int, int, int],
        dtype: np.dtype,
    ):
        self.videopath = videopath
        self.fps = fps
        self.queue = queue
        self.shared_memory_names = shared_memory_names
        self.shape = shape
        self.dtype = dtype

        self.shared_pointer = 0

    def __call__(
        self,
    ):
        with self:
            frame_no = 0
            frame_time = 0.0
            while True:
                (success, frame) = self.video_capture.read(self.shared_arrays[self.shared_pointer])
                if not success:
                    break

                self.queue.put(
                    (self.shared_memory_names[self.shared_pointer], frame_no, frame_time),
                    timeout=TIMEOUT,
                )

                frame_no += 1
                frame_time = frame_no / self.fps

                if self.shared_pointer == 0:
                    self.shared_pointer = len(self.shared_memory_names) - 1
                else:
                    self.shared_pointer -= 1

    def __enter__(self) -> None:
        self.video_capture = cv2.VideoCapture(self.videopath, cv2.CAP_ANY)

        self.shared_memory = tuple(SharedMemory(name, False) for name in self.shared_memory_names)
        self.shared_arrays = tuple(
            np.ndarray(self.shape, dtype=self.dtype, buffer=shared_memory.buf) for shared_memory in self.shared_memory
        )

    def __exit__(self, _type, value, traceback) -> None:
        self.video_capture.release()

        for mem in self.shared_memory:
            mem.close()
        self.shared_memory = ()
        self.shared_arrays = ()

        # sent end sentinel
        self.queue.put(None, timeout=TIMEOUT)


class FrameConsumer:
    queue: Queue
    shape: tuple[int, int, int]
    dtype: np.dtype
    shared_memory: dict[str, SharedMemory]
    shared_array: dict[str, np.ndarray]
    frame_no: int | None

    def __init__(
        self,
        queue: Queue,
        shape: tuple[int, int, int],
        dtype: np.dtype,
    ):
        self.queue = queue
        self.shape = shape
        self.dtype = dtype
        self.shared_memory = {}
        self.shared_array = {}

    def __call__(self):
        while item := self.queue.get(timeout=TIMEOUT):
            shared_memory_name, frame_no, frame_time = item
            self.frame_no = frame_no

            if shared_memory_name not in self.shared_memory:
                self.shared_memory[shared_memory_name] = SharedMemory(name=shared_memory_name, create=False)
                self.shared_array[shared_memory_name] = np.ndarray(
                    self.shape, dtype=self.dtype, buffer=self.shared_memory[shared_memory_name].buf
                )
            self.process_frame(self.shared_array[shared_memory_name])

        self.cleanup()
        self.queue.close()

        for shared_memory_item in self.shared_memory.values():
            shared_memory_item.close()

    def process_frame(self, frame) -> None:
        pass

    def cleanup(self) -> None:
        for mem in self.shared_memory.values():
            mem.close()


class FrameConsumerProducer:
    queue_in: Queue
    queue_out: Queue
    shape_in: tuple[int, int, int]
    shape_out: tuple[int, int, int]
    dtype_int: np.dtype
    dtype_out: np.dtype
    shared_memory_in = dict[str, SharedMemory]
    shared_array_in = dict[str, np.ndarray]
    shared_memory_names_out: tuple[str, ...]
    shared_memory_out: tuple[SharedMemory, ...]
    shared_array_out: tuple[np.ndarray]
    shared_pointer: int
    frame_no: int | None

    def __init__(
        self,
        queue_in: Queue,
        queue_out: Queue,
        shared_memory_names_out: tuple[str, ...],
        shape_in: tuple[int, int, int],
        shape_out: tuple[int, int, int],
        dtype_in: np.dtype,
        dtype_out: np.dtype,
    ):
        self.queue_in = queue_in
        self.queue_out = queue_out
        self.shared_memory_names_out = shared_memory_names_out
        self.shape_in = shape_in
        self.shape_out = shape_out
        self.dtype_in = dtype_in
        self.dtype_out = dtype_out
        self.shared_memory_in = {}
        self.shared_array_in = {}
        self.shared_pointer = 0

    def __call__(self):
        with self:
            while item := self.queue_in.get(timeout=TIMEOUT):
                shared_memory_name_in, frame_no, frame_time = item
                self.frame_no = frame_no

                if shared_memory_name_in not in self.shared_memory_in:
                    self.shared_memory_in[shared_memory_name_in] = SharedMemory(
                        name=shared_memory_name_in, create=False
                    )
                    self.shared_array_in[shared_memory_name_in] = np.ndarray(
                        self.shape_in, dtype=self.dtype_in, buffer=self.shared_memory_in[shared_memory_name_in].buf
                    )

                has_output = self.process_frame(
                    self.shared_array_in[shared_memory_name_in],
                    self.shared_array_out[self.shared_pointer],
                )

                if has_output:
                    self.queue_out.put(
                        (self.shared_memory_names_out[self.shared_pointer], frame_no, frame_time),
                        timeout=TIMEOUT,
                    )

                    if self.shared_pointer == 0:
                        self.shared_pointer = len(self.shared_memory_names_out) - 1
                    else:
                        self.shared_pointer -= 1

    def __enter__(self) -> None:
        self.shared_memory_out = tuple(SharedMemory(name, False) for name in self.shared_memory_names_out)
        self.shared_array_out = tuple(
            np.ndarray(self.shape_out, dtype=self.dtype_out, buffer=shared_memory.buf)
            for shared_memory in self.shared_memory_out
        )

    def __exit__(self, _type, value, traceback) -> None:
        for mem in self.shared_memory_in.values():
            mem.close()
        self.shared_memory_in = ()
        self.shared_arrays_in = ()

        self.queue_in.close()
        # sent end sentinel
        self.queue_out.put(None, timeout=TIMEOUT)

    def process_frame(self, frame_in: np.ndarray, frame_out: np.ndarray) -> bool:
        frame_out[:] = frame_in[:]
        return True


class SharedMemoryQueueResources:
    queue: Queue
    _shared_memory: tuple[SharedMemory, ...]

    def __init__(self, nbytes: int, size: int = 2):
        if size < 2:  # noqa: PLR2004
            msg = "size < 2"
            raise ValueError(msg)
        self.queue = Queue(size - 1)
        self._shared_memory = tuple(SharedMemory(create=True, size=nbytes) for _ in range(size))
        self.shared_memory_names = tuple(m.name for m in self._shared_memory)

    def __enter__(self) -> None:
        pass

    def __exit__(self, _type, value, traceback) -> None:
        self.close()

    def close(self) -> None:
        for mem in self._shared_memory:
            mem.close()
            mem.unlink()


def generate_from_queue_source(queue: Queue, timeout: float | None = None) -> Generator[SharedMemory, None, None]:
    memory_map = {}
    try:
        while item := queue.get(timeout=timeout):
            name = item[0]
            if shared_memory := memory_map[name]:
                pass
            else:
                shared_memory = SharedMemory(name, create=False)
                memory_map[name] = shared_memory

            yield shared_memory, *item[1:]
    finally:
        for shared_memory in memory_map.values():
            shared_memory.close()
