import logging
from collections.abc import Iterable
from contextlib import ExitStack
from csv import DictWriter
from multiprocessing import JoinableQueue, Queue
from subprocess import Popen

import cv2
import ffmpeg
import numpy as np

from camvidlog.frameinfo import Colourspace, FrameInfo, VideoFileStats
from camvidlog.queues import (
    SharedMemoryConsumer,
    SharedMemoryProducer,
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
        self.frame_no = 0

    def setup(self) -> None:
        pass

    def close(self) -> None:
        pass

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
                        frame, extras = content
                        sucess, frame_no, frame_time = self.generate_frame_into(frame)
                        extras.append(frame_no)
                        extras.append(frame_time)
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
    info_input: FrameInfo
    queue_in: JoinableQueue
    queues_out: tuple[JoinableQueue, ...]

    def __init__(self, info_input: FrameInfo, queue_in: JoinableQueue, queues_out: Iterable[JoinableQueue]):
        self.info_input = info_input
        self.queue_in = queue_in
        self.queues_out = tuple(queues_out)

    def __call__(self):
        try:
            self.setup()
            consumer = SharedMemoryConsumer(self.info_input, self.queue_in)
            producers = [SharedMemoryProducer(self.info_input, q) for q in self.queues_out]
            with ExitStack() as stack:
                [stack.enter_context(producer) for producer in producers]
                for item_in, items_out in zip(consumer, zip(producers, strict=True), strict=True):
                    with item_in as content:
                        frame_in, frame_no, frame_time, *extras_in = content
                        logger.debug(f"{self} processing {frame_no}")
                        for i, item_out in enumerate(items_out):
                            with item_out as product:
                                frame_out, *extras_out = product
                                logger.debug(f"{self} processing {frame_no} to {i}")
                                np.copyto(frame_out, frame_in)
        finally:
            self.close()

    def setup(self) -> None:
        pass

    def close(self) -> None:
        pass


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
