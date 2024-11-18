import logging
from collections.abc import Iterable
from csv import DictWriter
from multiprocessing import JoinableQueue, Queue
from multiprocessing.shared_memory import SharedMemory
from subprocess import Popen

import cv2
import ffmpeg
import numpy as np

from camvidlog.frameinfo import Colourspace, FrameInfo, VideoFileStats
from camvidlog.queues import (
    SharedMemoryConsumer,
    SharedMemoryProducer,
    SharedMemoryQueueManager,
    SharedMemoryQueueResources,
)

logger = logging.getLogger(__name__)

TIMEOUT = 600.0


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


# TODO detect if a "colour" video is infact greyscale


class FrameProducer:
    info_output: FrameInfo
    queue: JoinableQueue

    def __init__(self, info_input: FrameInfo, queue: JoinableQueue):
        self.info_input = info_input
        self.queue = queue

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
            self.setup()
            with SharedMemoryProducer(self.info_input, self.queue) as producer:
                for item in producer:
                    with item as content:
                        frame, frame_no, frame_time, *extras = content
                        self.frame_no = frame_no
                        self.frame_time = frame_time
                        sucess, *extras = self.generate_frame_into(frame)
                        if not sucess:
                            break
        finally:
            self.close()


class FileReader(FrameProducer):
    filename: str
    _video_capture: cv2.VideoCapture | None = None
    _video_file_stats: VideoFileStats | None = None
    info_output: FrameInfo
    frame_no: int = 0
    frame_time: float = 0.0

    def __init__(self, filename: str, queue: JoinableQueue):
        self.filename = filename
        self.info_output = FrameInfo(
            self.video_file_stats.x, self.video_file_stats.y, self.video_file_stats.colourspace
        )
        super().__init__(self.info_output, queue)

    @property
    def video_file_stats(self):
        if self._video_file_stats is None:
            self._video_file_stats = peek_in_file(self.filename)
        return self._video_file_stats

    def setup(self) -> None:
        super().setup()
        self._video_capture = cv2.VideoCapture(self.filename, cv2.CAP_ANY)

    def close(self) -> None:
        super().close()
        if self._video_capture is not None:
            self._video_capture.release()

    def generate_frame_into(self, array: np.ndarray) -> tuple[bool, int | None, float | None]:
        (success, _) = self._video_capture.read(array)
        if not success:
            return False, None, None
        else:
            self.frame_no += 1
            self.frame_time = self.frame_no / self.video_file_stats.fps
            return True, self.frame_no, self.frame_time


class FFMPEGReader(FrameProducer):
    filename: str
    reader: Popen | None = None
    _video_file_stats: VideoFileStats | None = None
    info_output: FrameInfo
    frame_no: int = 0
    frame_time: float = 0.0

    def __init__(self, filename: str, queue: JoinableQueue):
        self.filename = filename
        self.info_output = FrameInfo(
            self.video_file_stats.x, self.video_file_stats.y, self.video_file_stats.colourspace
        )
        super().__init__(self.info_output, queue)

    @property
    def video_file_stats(self):
        if self._video_file_stats is None:
            self._video_file_stats = peek_in_file(self.filename)
        return self._video_file_stats

    def setup(self) -> None:
        self.reader = (
            ffmpeg.input(self.filename, hwaccel="cuda")
            .output("pipe:", format="rawvideo", pix_fmt="rgb24")
            .run_async(pipe_stdout=True, quiet=True)
        )

    def close(self) -> None:
        super().close()

    def generate_frame_into(self, array: np.ndarray) -> tuple[bool, int | None, float | None]:
        if self.reader.poll() is not None:
            return False, None, None

        in_bytes = self.reader.stdout.read(self.info_output.nbytes)
        if not in_bytes:
            return False, None, None

        array_buffer = np.frombuffer(in_bytes, np.uint8).reshape(
            [self.info_output.y, self.info_output.x, 3 if self.info_output.colourspace == Colourspace.RGB else 1]
        )
        np.copyto(array, array_buffer)

        self.frame_no += 1
        self.frame_time = self.frame_no / self.video_file_stats.fps
        return True, self.frame_no, self.frame_time


class FrameConsumer:
    info_input: FrameInfo
    consumer: SharedMemoryConsumer
    frame_no: int | None
    frame_time: float | None

    def __init__(self, info_input: FrameInfo, queue: JoinableQueue):
        self.info_input = info_input
        self.queue = queue

    def __call__(self):
        try:
            self.setup()
            consumer = SharedMemoryConsumer(self.info_input, self.queue)
            for item in consumer:
                with item as content:
                    frame, frame_no, frame_time, *extras = content
                    self.frame_no = frame_no
                    self.frame_time = frame_time
                    self.process_frame(frame)
        finally:
            self.close()

    def setup(self) -> None:
        pass

    def process_frame(self, frame) -> None:
        pass

    def generate_frame_into(self, array: np.ndarray) -> tuple[bool, int | None, float | None]:
        raise NotImplementedError

    def close(self) -> None:
        pass


class FrameConsumerProducer:
    info_input: FrameInfo
    queue_in: JoinableQueue
    queue_out: JoinableQueue
    frame_no: int | None
    frame_time: float | None

    def __init__(self, info_input: FrameInfo, queue_in: JoinableQueue, queue_out: JoinableQueue):
        self.info_input = info_input
        self.queue_in = queue_in
        self.queue_out = queue_out

    def setup(self) -> None:
        pass

    def process_frame(self, frame_in: np.ndarray, frame_out: np.ndarray) -> bool:
        np.copyto(frame_out, frame_in)
        return True

    def close(self) -> None:
        pass

    def __call__(self):
        try:
            self.setup()
            consumer = SharedMemoryConsumer(self.info_input, self.queue_in)
            with SharedMemoryProducer(self.info_input, self.queue_out) as producer:
                for item_in, item_out in zip(consumer, producer, strict=False):
                    with item_out as product:
                        with item_in as content:
                            frame_in, frame_no, frame_time, *extras_in = content
                            frame_out, *extras_out = product
                            self.frame_no = frame_no
                            self.frame_time = frame_time
                            # TODO control if output is propegated to queue or not
                            item_out.propegate = self.process_frame(frame_in, frame_out)
                            extras_out.extend(frame_no, frame_time, *extras_in)
        finally:
            self.close()


class FrameCopier:
    queue_resourcess: list[SharedMemoryQueueResources]

    def __init__(self, info_input: FrameInfo, queue_in: JoinableQueue, queues_out: Iterable[JoinableQueue]):
        self.info_input = FrameInfo

        self.info_outputs = [self.info_output]
        self.info_output = None
        self.queue_resourcess = [self.queue_resources]
        self.queue_resources = None
        for _ in range(1, copy_number):
            queue_resources = queue_manager.make_queue(nbytes)
            self.queue_resourcess.append(queue_resources)
            self.info_outputs.append(FrameInfo(queue_resources.queue, x, y, colourspace))

    def _get_nbytes(self) -> int:
        return self.info_input.nbytes

    def _get_x_y(self) -> tuple[int, int]:
        return (self.info_input.x, self.info_input.y)

    def _get_colourspace(self) -> Colourspace:
        return self.info_input.colourspace

    def __call__(self):
        try:
            self.setup()
            shared_pointer = 0
            shared_memory_in: dict[str, SharedMemory] = {}
            shared_array_in: dict[str, np.ndarray] = {}

            shared_memory_out: tuple[tuple[SharedMemory, ...]] = tuple(
                tuple(SharedMemory(name, False) for name in queue_resource.shared_memory_names)
                for queue_resource in self.queue_resourcess
            )
            shared_array_out: tuple[tuple[np.ndarray, ...]] = tuple(
                tuple(
                    np.ndarray(self.info_outputs[0].shape, dtype=np.uint8, buffer=shared_memory.buf)
                    for shared_memory in shared_memory_out_inner
                )
                for shared_memory_out_inner in shared_memory_out
            )

            while item := self.info_input.queue.get(timeout=TIMEOUT):
                shared_memory_name_in, frame_no, frame_time = item
                self.frame_no = frame_no

                if shared_memory_name_in not in shared_memory_in:
                    shared_memory_in[shared_memory_name_in] = SharedMemory(name=shared_memory_name_in, create=False)
                    shared_array_in[shared_memory_name_in] = np.ndarray(
                        self.info_input.shape, dtype=np.uint8, buffer=shared_memory_in[shared_memory_name_in].buf
                    )

                self.info_input.queue.task_done()
                logger.debug(f"{self} consumed {frame_no:4d}")

                for shared_array_out_inner in shared_array_out:
                    np.copyto(shared_array_out_inner[shared_pointer], shared_array_in[shared_memory_name_in])

                for info_output, queue_resources in zip(self.info_outputs, self.queue_resourcess, strict=False):
                    info_output.queue.put(
                        (queue_resources.shared_memory_names[shared_pointer], frame_no, frame_time),
                        timeout=TIMEOUT,
                    )

                if shared_pointer == 0:
                    shared_pointer = len(self.queue_resourcess[0].shared_memory_names) - 1
                else:
                    shared_pointer -= 1
            # "done" the sentinel
            self.info_input.queue.task_done()
        finally:
            self.close()

            for mem in shared_memory_in.values():
                mem.close()
            for mems in shared_memory_out:
                for mem in mems:
                    mem.close()

            self.info_input.queue.close()
            # sent end sentinel
            for info_output in self.info_outputs:
                info_output.queue.put(None, timeout=TIMEOUT)


class DataRecorder:
    queue: Queue
    sentinels_max: int
    sentinels_current: int
    metrics: dict[int, dict[str, int | float]]
    columns: list[str]

    def __init__(self, queue: Queue, sentinels_max: int, outfilename: str):
        self.queue = queue
        self.sentinels_max = sentinels_max
        self.sentinels_current = 0
        self.metrics = {}
        self.outfilename = outfilename
        self.columns = ["frame_no"]

    def register(self, columns: Iterable[str]) -> Queue:
        self.columns.extend(column for column in columns if column not in self.columns)
        return self.queue

    def __call__(self):
        running = True
        with open(self.outfilename, "w") as outfile:
            dict_writer = DictWriter(outfile, self.columns)
            dict_writer.writeheader()
            while running:
                item = self.queue.get(timeout=TIMEOUT)
                if item is None:
                    self.sentinels_current += 1
                    if self.sentinels_current >= self.sentinels_max:
                        running = False
                else:
                    frame_no, metrics = item
                    row = dict(metrics)
                    row["frame_no"] = frame_no
                    dict_writer.writerow(row)
