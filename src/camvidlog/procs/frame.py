from multiprocessing import Queue

import cv2
import numpy as np

from camvidlog.procs.basics import FrameConsumer, FrameConsumerProducer


class SaveToFile(FrameConsumer):
    filename: str
    fps: float
    out: cv2.VideoWriter | None

    def __init__(self, filename: str, fps: float, queue: Queue, shape: tuple[int, int, int], dtype: np.dtype):
        super().__init__(queue=queue, shape=shape, dtype=dtype)
        self.filename = filename
        self.fps = fps
        self.out = None

    def process_frame(self, frame) -> None:
        if not self.out:
            self.out = cv2.VideoWriter(
                self.filename,
                # cv2.VideoWriter_fourcc(*"MJPG"),
                cv2.VideoWriter_fourcc(*"XVID"),
                # cv2.VideoWriter_fourcc(*"X264"), # .mp4
                self.fps,
                (self.shape[1], self.shape[0]),
                isColor=(self.shape[2] == 3),  # noqa: PLR2004
            )
        self.out.write(frame)

    def cleanup(self) -> None:
        if self.out:
            self.out.release()


class BackgroundSubtractorMOG2(FrameConsumerProducer):
    background_subtractor: cv2.BackgroundSubtractorMOG2

    def __init__(
        self,
        queue_in: Queue,
        queue_out: Queue,
        shared_memory_names_out: tuple[str, ...],
        shape: tuple[int, int, int],
        dtype: np.dtype,
        history: int = 500,
        var_threshold=16,
    ):
        super().__init__(
            queue_in=queue_in,
            queue_out=queue_out,
            shared_memory_names_out=shared_memory_names_out,
            shape_in=shape,
            shape_out=shape,
            dtype_in=dtype,
            dtype_out=dtype,
        )
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history, detectShadows=False, varThreshold=var_threshold
        )

    def process_frame(self, frame_in, frame_out) -> bool:
        fgmask = self.background_subtractor.apply(frame_in)
        cv2.cvtColor(fgmask, cv2.COLOR_GRAY2RGB, frame_out)
        return True


class BackgroundSubtractorKNN(FrameConsumerProducer):
    background_subtractor: cv2.BackgroundSubtractorKNN

    def __init__(
        self,
        queue_in: Queue,
        queue_out: Queue,
        shared_memory_names_out: tuple[str, ...],
        shape: tuple[int, int, int],
        dtype: np.dtype,
        history: int = 500,
        dist2_threshold: float = 400.0,
    ):
        super().__init__(
            queue_in=queue_in,
            queue_out=queue_out,
            shared_memory_names_out=shared_memory_names_out,
            shape_in=shape,
            shape_out=shape,
            dtype_in=dtype,
            dtype_out=dtype,
        )
        self.background_subtractor = cv2.createBackgroundSubtractorKNN(
            history=history, detectShadows=False, dist2Threshold=dist2_threshold
        )

    def process_frame(self, frame_in, frame_out) -> bool:
        fgmask = self.background_subtractor.apply(frame_in)
        cv2.cvtColor(fgmask, cv2.COLOR_GRAY2RGB, frame_out)
        return True


class Rescaler(FrameConsumerProducer):
    res: tuple[int, int]
    fps_ratio: float

    def __init__(
        self,
        queue_in: Queue,
        queue_out: Queue,
        shared_memory_names_out: tuple[str, ...],
        shape_in: tuple[int, int, int],
        shape_out: tuple[int, int, int],
        dtype_in: np.dtype,
        dtype_out: np.dtype,
        fps_in: int,
        fps_out: int,
    ):
        super().__init__(
            queue_in=queue_in,
            queue_out=queue_out,
            shared_memory_names_out=shared_memory_names_out,
            shape_in=shape_in,
            shape_out=shape_out,
            dtype_in=dtype_in,
            dtype_out=dtype_out,
        )
        self.res = (shape_out[1], shape_out[0])  # needs to be x,y but shape is y,x
        if fps_out > fps_in:
            msg = "fps_out cannot be greater than fps_in"
            raise ValueError(msg)
        self.fps_ratio = fps_in / fps_out

    def process_frame(self, frame_in, frame_out) -> bool:
        if self.frame_no % self.fps_ratio < 1.0:
            cv2.resize(frame_in, self.res, frame_out)
            return True
        else:
            # skip frame
            return False
