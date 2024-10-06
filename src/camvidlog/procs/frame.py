import logging
from multiprocessing import Queue
from subprocess import Popen

import cv2
import ffmpeg
import numpy as np

from camvidlog.procs.basics import Colourspace, DataRecorder, FrameConsumer, FrameConsumerProducer, FrameQueueInfoOutput
from camvidlog.queues import SharedMemoryQueueManager

logger = logging.getLogger(__name__)


class SaveToFile(FrameConsumer):
    filename: str
    fps: float
    out: cv2.VideoWriter | None

    def __init__(self, filename: str, fps: float, info_input: FrameQueueInfoOutput):
        super().__init__(info_input=info_input)
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
                (self.info_input.x, self.info_input.y),
                isColor=self.info_input.colourspace != Colourspace.greyscale,
            )
        self.out.write(frame)
        logger.debug(f"Wrote {self.frame_no:4f} to {self.filename}")

    def close(self) -> None:
        if self.out:
            self.out.release()


class FFMPEGToFile(FrameConsumer):
    filename: str
    fps: float
    out: Popen | None = None

    def __init__(self, filename: str, fps: float, info_input: FrameQueueInfoOutput):
        super().__init__(info_input=info_input)
        self.filename = filename
        self.fps = fps
        self.out = None

    def process_frame(self, frame) -> None:
        # TODO handle grayscale
        if not self.out:
            if self.info_input.colourspace == Colourspace.RGB:
                pix_fmt = "rgb24"
            elif self.info_input.colourspace == Colourspace.greyscale:
                pix_fmt = "gray"
            else:
                msg = f"unsupported colourspace {self.info_input.colourspace}"
                raise ValueError(msg)
            self.out = (
                ffmpeg.input(
                    "pipe:",
                    format="rawvideo",
                    pix_fmt=pix_fmt,
                    r=self.fps,
                    s=f"{self.info_input.x}x{self.info_input.y}",
                    hwaccel="cuda",
                )
                .output(self.filename, pix_fmt="yuv420p")
                .overwrite_output()
                .run_async(
                    pipe_stdin=True,
                    quiet=True,
                )
            )
        self.out.stdin.write(frame.astype(np.uint8).tobytes())

        logger.debug(f"Wrote {self.frame_no:4f} to {self.filename}")

    def close(self):
        if self.out:
            self.out.stdin.close()
            self.out.wait()


class Rescaler(FrameConsumerProducer):
    res: tuple[int, int]
    fps_ratio: float

    def __init__(
        self,
        x: int,
        y: int,
        fps_in: int,
        fps_out: int,
        info_input: FrameQueueInfoOutput,
        queue_manager: SharedMemoryQueueManager,
    ):
        self.res = (x, y)
        super().__init__(info_input=info_input, queue_manager=queue_manager)
        if fps_out > fps_in:
            msg = "fps_out cannot be greater than fps_in"
            raise ValueError(msg)
        self.fps_ratio = fps_in / fps_out

    def process_frame(self, frame_in, frame_out) -> bool:
        if self.frame_no % self.fps_ratio < 1.0:
            cv2.resize(frame_in, self.res, frame_out)  # , interpolation=cv2.INTER_LANCZOS4)
            return True
        else:
            # skip frame
            return False

    def _get_nbytes(self) -> int:
        return self.res[0] * self.res[1] * 3  # TODO not assume colour

    def _get_x_y(self) -> tuple[int, int]:
        return (self.res[0], self.res[1])
