import logging
from csv import DictWriter
from dataclasses import dataclass
from enum import Enum
from multiprocessing import Queue
from multiprocessing.shared_memory import SharedMemory
from typing import Self

import cv2
import numpy as np

from camvidlog.procs.queues import SharedMemoryQueueManager, SharedMemoryQueueResources

logger = logging.getLogger(__name__)

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

    @property
    def area(self) -> int:
        return self.x * self.y

    @property
    def nbytes(self) -> int:
        return (
            self.x * self.y * (1 if self.colourspace == Colourspace.greyscale else 3) * (np.iinfo(np.uint8).bits // 8)
        )


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


class FrameProducer:
    queue_resources: SharedMemoryQueueResources
    info_output: FrameQueueInfoOutput

    def __init__(self, *, queue_manager: SharedMemoryQueueManager, **kwargs):
        super().__init__(**kwargs)
        nbytes = self._get_nbytes()
        self.queue_resources = queue_manager.make_queue(nbytes)
        x, y = self._get_x_y()
        colourspace = self._get_colourspace()
        self.info_output = FrameQueueInfoOutput(self.queue_resources.queue, x, y, colourspace)

    def _get_nbytes(self) -> int:
        raise NotImplementedError

    def _get_x_y(self) -> tuple[int, int]:
        raise NotImplementedError

    def _get_colourspace(self) -> Colourspace:
        raise NotImplementedError

    def _get_shape(self) -> tuple[int, int, int]:
        # Y,X to match openCV
        x, y = self._get_x_y()
        return (y, x, 1 if self._get_colourspace() == Colourspace.greyscale else 3)

    def setup(self) -> None:
        pass

    def close(self) -> None:
        # sent end sentinel
        self.info_output.queue.put(None, timeout=TIMEOUT)

    def generate_frame_into(self, array: np.ndarray) -> tuple[bool, int | None, float | None]:
        raise NotImplementedError

    def __call__(
        self,
    ):
        try:
            # setup locals to avoid accidental pickle
            shared_memory = ()
            shared_memory = tuple(SharedMemory(name, False) for name in self.queue_resources.shared_memory_names)
            shared_arrays = tuple(
                np.ndarray(self._get_shape(), dtype=np.uint8, buffer=shared_memory.buf)
                for shared_memory in shared_memory
            )
            shared_pointer = 0

            self.setup()

            while True:
                success, frame_no, frame_time = self.generate_frame_into(shared_arrays[shared_pointer])
                if not success:
                    break
                logger.debug(f"{self} generated {frame_no:4d}")

                self.info_output.queue.put(
                    (self.queue_resources.shared_memory_names[shared_pointer], frame_no, frame_time),
                    timeout=TIMEOUT,
                )

                if shared_pointer == 0:
                    shared_pointer = len(self.queue_resources.shared_memory_names) - 1
                else:
                    shared_pointer -= 1
        finally:
            self.close()
            for mem in shared_memory:
                mem.close()


class FileReader(FrameProducer):
    filename: str
    _video_file_stats: VideoFileStats | None = None
    info_output: FrameQueueInfoOutput
    video_capture: cv2.VideoCapture | None = None
    frame_no: int = 0
    frame_time: float = 0.0

    def __init__(self, *, filename: str, **kwargs):
        self.filename = filename
        super().__init__(**kwargs)

    @classmethod
    def from_file(cls, filename: str, queue: Queue, shared_memory_names: tuple[str, ...]) -> Self:
        vidstats = peek_in_file(filename)
        return cls(filename, vidstats.fps, queue, shared_memory_names, vidstats.x, vidstats.y, vidstats.colourspace)

    @property
    def _shape(self):
        shape = (self.info_output.y, self.info_output.x)
        shape = (*shape, 3) if self.info_output.colourspace == Colourspace.RGB else shape
        return shape

    @property
    def video_file_stats(self):
        if self._video_file_stats is None:
            self._video_file_stats = peek_in_file(self.filename)
        return self._video_file_stats

    def _get_nbytes(self) -> int:
        return self.video_file_stats.nbytes

    def _get_x_y(self) -> tuple[int, int]:
        return self.video_file_stats.x, self.video_file_stats.y

    def _get_colourspace(self) -> Colourspace:
        return self.video_file_stats.colourspace

    def setup(self) -> None:
        super().setup()
        self.video_capture = cv2.VideoCapture(self.filename, cv2.CAP_ANY)

    def close(self) -> None:
        super().close()
        if self.video_capture is not None:
            self.video_capture.release()

    def generate_frame_into(self, array: np.ndarray) -> tuple[bool, int | None, float | None]:
        (success, _) = self.video_capture.read(array)
        if not success:
            return False, None, None
        else:
            self.frame_no += 1
            self.frame_time = self.frame_no / self.video_file_stats.fps
            return True, self.frame_no, self.frame_time


class FrameConsumer:
    info_input: FrameQueueInfoOutput
    frame_no: int | None

    def __init__(self, *, info_input: FrameQueueInfoOutput, **kwargs):
        self.info_input = info_input
        super().__init__(**kwargs)

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
                logger.debug(f"{self} consumed {frame_no:4d}")
        finally:
            self.close()

            for shared_memory_item in shared_memory.values():
                shared_memory_item.close()

    def setup(self) -> None:
        pass

    def process_frame(self, frame) -> None:
        pass

    def close(self) -> None:
        self.info_input.queue.close()


class FrameConsumerProducer(FrameConsumer, FrameProducer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _get_nbytes(self) -> int:
        return self.info_input.nbytes

    def _get_x_y(self) -> tuple[int, int]:
        return (self.info_input.x, self.info_input.y)

    def _get_colourspace(self) -> Colourspace:
        return self.info_input.colourspace

    def __call__(self):
        # optimize avoid object dereferencing
        queue_out = self.info_output.queue
        shared_memory_names = self.queue_resources.shared_memory_names
        try:
            self.setup()
            shared_pointer = 0
            shared_memory_in: dict[str, SharedMemory] = {}
            shared_array_in: dict[str, np.ndarray] = {}

            shared_memory_out: tuple[SharedMemory, ...] = tuple(
                SharedMemory(name, False) for name in shared_memory_names
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
                logger.debug(f"{self} consumed {frame_no:4d}")

                if has_output:
                    queue_out.put(
                        (shared_memory_names[shared_pointer], frame_no, frame_time),
                        timeout=TIMEOUT,
                    )

                    if shared_pointer == 0:
                        shared_pointer = len(shared_memory_names) - 1
                    else:
                        shared_pointer -= 1
        finally:
            self.close()

            for mem in shared_memory_in.values():
                mem.close()
            for mem in shared_memory_out:
                mem.close()

            self.info_input.queue.close()
            # sent end sentinel
            self.info_output.queue.put(None, timeout=TIMEOUT)

    def process_frame(self, frame_in: np.ndarray, frame_out: np.ndarray) -> bool:
        np.copyto(frame_out, frame_in)
        return True


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
