from collections.abc import Generator
from csv import DictWriter
from dataclasses import dataclass
from enum import Enum
from multiprocessing import Queue
from multiprocessing.shared_memory import SharedMemory

import cv2
import numpy as np

TIMEOUT = 180.0


class Resolution(Enum):
    # Y,X to match openCV
    VGA = (480, 640)
    SD = (720, 1280)
    HD = (1080, 1920)  # 2k
    UHD = (2160, 3840)  # 4K


class Colourspace(Enum):
    RGB = "rgb"
    greyscale = "greyscale"


@dataclass
class FrameQueueInfoOutput:
    queue: Queue
    x: int
    y: int
    colourspace: Colourspace

    @property
    def shape(self) -> tuple[int, int, int]:
        # Y,X to match openCV
        return (self.y, self.x, 1 if self.colourspace == Colourspace.greyscale else 3)


@dataclass
class VideoFileStats:
    filename: str
    fps: float
    x: int
    y: int
    colourspace: Colourspace

    @property
    def shape(self) -> tuple[int, int, int]:
        # Y,X to match openCV
        return (self.y, self.x, 1 if self.colourspace == Colourspace.greyscale else 3)

    @property
    def nbytes(self) -> int:
        if self.colourspace == Colourspace.greyscale:
            return self.x * self.y
        else:
            return self.x * self.y * 3


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
        colourspace = Colourspace.greyscale if frame.shape[2] == 1 else Colourspace.RGB
        return VideoFileStats(filename=filename, fps=fps, x=x, y=y, colourspace=colourspace)
    finally:
        if video_capture:
            video_capture.release()
            video_capture = None


class FileReader:
    videopath: str
    shared_memory_names: tuple[str, ...]
    fps: float

    info_output: FrameQueueInfoOutput

    def __init__(
        self,
        videopath: str,
        fps: float,
        queue: Queue,
        shared_memory_names: tuple[str, ...],
        x: int,
        y: int,
        colourspace: Colourspace,
    ):
        self.videopath = videopath
        self.fps = fps
        self.shared_memory_names = shared_memory_names

        self.info_output = FrameQueueInfoOutput(queue, x, y, colourspace)

    @property
    def _shape(self):
        shape = (self.info_output.y, self.info_output.x)
        shape = (*shape, 3) if self.info_output.colourspace == Colourspace.RGB else shape
        return shape

    def __call__(
        self,
    ):
        video_capture = None
        shared_memory = ()
        try:
            video_capture = cv2.VideoCapture(self.videopath, cv2.CAP_ANY)
            shared_memory = tuple(SharedMemory(name, False) for name in self.shared_memory_names)
            shared_arrays = tuple(
                np.ndarray(self._shape, dtype=np.uint8, buffer=shared_memory.buf) for shared_memory in shared_memory
            )
            shared_pointer = 0

            frame_no = 0
            frame_time = 0.0
            while True:
                (success, _) = video_capture.read(shared_arrays[shared_pointer])
                if not success:
                    break

                self.info_output.queue.put(
                    (self.shared_memory_names[shared_pointer], frame_no, frame_time),
                    timeout=TIMEOUT,
                )

                frame_no += 1
                frame_time = frame_no / self.fps

                if shared_pointer == 0:
                    shared_pointer = len(self.shared_memory_names) - 1
                else:
                    shared_pointer -= 1
        finally:
            if video_capture:
                video_capture.release()

            for mem in shared_memory:
                mem.close()

            # sent end sentinel
            self.info_output.queue.put(None, timeout=TIMEOUT)


class FrameConsumer:
    info_input: FrameQueueInfoOutput
    frame_no: int | None

    def __init__(self, info_input: FrameQueueInfoOutput):
        self.info_input = info_input
        print(f"{self} {info_input}")

    def __call__(self):
        try:
            self.setup()
            shared_memory: dict[str, SharedMemory] = {}
            shared_array: dict[str, np.ndarray] = {}

            while item := self.info_input.queue.get(timeout=TIMEOUT):
                shared_memory_name, frame_no, frame_time = item
                self.frame_no = frame_no

                if shared_memory_name not in shared_memory:
                    shared_memory[shared_memory_name] = SharedMemory(name=shared_memory_name, create=False)
                    shared_array[shared_memory_name] = np.ndarray(
                        self.info_input.shape, dtype=np.uint8, buffer=shared_memory[shared_memory_name].buf
                    )
                self.process_frame(shared_array[shared_memory_name])
        finally:
            self.cleanup()
            self.info_input.queue.close()

            for shared_memory_item in shared_memory.values():
                shared_memory_item.close()

    def setup(self) -> None:
        pass

    def process_frame(self, frame) -> None:
        pass

    def cleanup(self) -> None:
        pass


class FrameConsumerProducer:
    shared_memory_names_out: tuple[str, ...]
    frame_no: int | None

    info_input: FrameQueueInfoOutput

    def __init__(
        self,
        info_input: FrameQueueInfoOutput,
        queue_out: Queue,
        shared_memory_names_out: tuple[str, ...],
        x: int,
        y: int,
        colourspace: Colourspace,
    ):
        self.info_input = info_input
        self.info_output = FrameQueueInfoOutput(queue_out, x, y, colourspace)
        self.shared_memory_names_out = shared_memory_names_out
        print(f"{self} {self.info_input} {self.info_output}")

    def __call__(self):
        try:
            self.setup()
            shared_pointer = 0
            shared_memory_in: dict[str, SharedMemory] = {}
            shared_array_in: dict[str, np.ndarray] = {}

            shared_memory_out: tuple[SharedMemory, ...] = tuple(
                SharedMemory(name, False) for name in self.shared_memory_names_out
            )
            shared_array_out: tuple[np.ndarray, ...] = tuple(
                np.ndarray(self.info_output.shape, dtype=np.uint8, buffer=shared_memory.buf)
                for shared_memory in shared_memory_out
            )

            while item := self.info_input.queue.get(timeout=TIMEOUT):
                shared_memory_name_in, frame_no, frame_time = item
                self.frame_no = frame_no

                if shared_memory_name_in not in shared_memory_in:
                    shared_memory_in[shared_memory_name_in] = SharedMemory(name=shared_memory_name_in, create=False)
                    shared_array_in[shared_memory_name_in] = np.ndarray(
                        self.info_input.shape, dtype=np.uint8, buffer=shared_memory_in[shared_memory_name_in].buf
                    )

                has_output = self.process_frame(
                    shared_array_in[shared_memory_name_in],
                    shared_array_out[shared_pointer],
                )

                if has_output:
                    self.info_output.queue.put(
                        (self.shared_memory_names_out[shared_pointer], frame_no, frame_time),
                        timeout=TIMEOUT,
                    )

                    if shared_pointer == 0:
                        shared_pointer = len(self.shared_memory_names_out) - 1
                    else:
                        shared_pointer -= 1
        finally:
            self.cleanup()

            for mem in shared_memory_in.values():
                mem.close()
            for mem in shared_memory_out:
                mem.close()

            self.info_input.queue.close()
            # sent end sentinel
            self.info_output.queue.put(None, timeout=TIMEOUT)

    def setup(self) -> None:
        pass

    def process_frame(self, frame_in: np.ndarray, frame_out: np.ndarray) -> bool:
        np.copyto(frame_out, frame_in)
        return True

    def cleanup(self) -> None:
        pass


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


class DataRecorder:
    queue: Queue
    sentinels_max: int
    sentinels_current: int
    metrics: dict[int, dict[str, int | float]]

    def __init__(self, queue: Queue, sentinels_max: int, outfilename: str):
        self.queue = queue
        self.sentinels_max = sentinels_max
        self.sentinels_current = 0
        self.metrics = {}
        self.outfilename = outfilename

    def __call__(self):
        running = True
        frame_max = 0
        while running:
            item = self.queue.get(timeout=TIMEOUT)
            if item is None:
                self.sentinels_current += 1
                if self.sentinels_current >= self.sentinels_max:
                    running = False
            else:
                frame_no, metric, value = item
                frame_max = max(frame_max, frame_no)
                if metric not in self.metrics:
                    self.metrics[metric] = {}
                self.metrics[metric][frame_no] = value

        with open(self.outfilename, "w") as outfile:
            dict_writer = DictWriter(outfile, ("frame_no", *tuple(self.metrics.keys())))
            dict_writer.writeheader()
            for i in range(frame_max):
                row = {}
                for metric in self.metrics:
                    if i in self.metrics[metric]:
                        row[metric] = self.metrics[metric][i]
                if row:
                    row["frame_no"] = i
                    dict_writer.writerow(row)
