import logging
from multiprocessing import Queue
from subprocess import Popen

import cv2
import ffmpeg
import numpy as np

from camvidlog.procs.basics import Colourspace, FrameConsumer, FrameConsumerProducer, FrameQueueInfoOutput
from camvidlog.procs.queues import SharedMemoryQueueManager

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
                raise ValueError(f"unsupported colourspace {self.info_input.colourspace}")
            self.out = (
                ffmpeg.input(
                    "pipe:",
                    format="rawvideo",
                    pix_fmt=pix_fmt,
                    r=self.fps,
                    s=f"{self.info_input.x}x{self.info_input.y}",
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


class BackgroundSubtractorMOG2(FrameConsumerProducer):
    _background_subtractor: cv2.BackgroundSubtractorMOG2 | None = None
    history: int
    var_threshold: int

    def __init__(
        self,
        info_input: FrameQueueInfoOutput,
        queue_manager: SharedMemoryQueueManager,
        history: int = 500,
        var_threshold=16,
    ):
        super().__init__(info_input=info_input, queue_manager=queue_manager)
        self.history = history
        self.var_threshold = var_threshold

        # always outputs greyscale
        self.info_output.colourspace = Colourspace.greyscale

    def process_frame(self, frame_in, frame_out) -> bool:
        if self._background_subtractor is None:
            self._background_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=self.history, detectShadows=False, varThreshold=self.var_threshold
            )
        self._background_subtractor.apply(frame_in, frame_out)
        return True


class BackgroundSubtractorKNN(FrameConsumerProducer):
    _background_subtractor: cv2.BackgroundSubtractorKNN | None = None
    history: int
    dist2_threshold: float

    def __init__(
        self,
        info_input: FrameQueueInfoOutput,
        queue_manager: SharedMemoryQueueManager,
        history: int = 500,
        dist2_threshold: float = 400.0,
    ):
        super().__init__(info_input=info_input, queue_manager=queue_manager)
        self.history = history
        self.dist2_threshold = dist2_threshold

        # always outputs greyscale
        self.info_output.colourspace = Colourspace.greyscale

    def process_frame(self, frame_in, frame_out) -> bool:
        if self._background_subtractor is None:
            self._background_subtractor = cv2.createBackgroundSubtractorKNN(
                history=self.history, detectShadows=False, dist2Threshold=self.dist2_threshold
            )
        self._background_subtractor.apply(frame_in, frame_out)
        return True


class BackgroundMaskDenoiser(FrameConsumerProducer):
    kernel_size: int
    frame_temp: None | np.ndarray = None

    def __init__(
        self,
        info_input: FrameQueueInfoOutput,
        queue_manager: SharedMemoryQueueManager,
        kernel_size: int = 3,
    ):
        super().__init__(info_input=info_input, queue_manager=queue_manager)
        self.kernel_size = kernel_size

    def process_frame(self, frame_in, frame_out) -> bool:
        # frigate uses bluring and thresholding
        # opencv tutorial uses morphology kernels
        kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)

        # To avoid allocating intermediate frame each time!
        if self.frame_temp is None:
            self.frame_temp = cv2.morphologyEx(frame_in, cv2.MORPH_OPEN, kernel)
        else:
            cv2.morphologyEx(frame_in, cv2.MORPH_OPEN, kernel, self.frame_temp)
        cv2.morphologyEx(self.frame_temp, cv2.MORPH_CLOSE, kernel, dst=frame_out)

        return True


class MaskStats(FrameConsumerProducer):
    queue_results: Queue
    prefix: str
    supplementary: dict[str, str]

    def __init__(
        self,
        info_input: FrameQueueInfoOutput,
        queue_manager: SharedMemoryQueueManager,
        queue_results: Queue,
        prefix: str = "",
        supplementary: dict[str, str] | None = None,
    ):
        super().__init__(info_input=info_input, queue_manager=queue_manager)
        self.queue_results = queue_results
        self.prefix = f"{prefix}." if prefix else ""
        self.supplementary = supplementary if supplementary else {}

    def process_frame(self, frame_in, frame_out) -> bool:
        metrics = {}
        mean = frame_in.mean()
        metrics["mask.mean"] = mean

        # TODO avoid allocation
        contours, _ = cv2.findContours(frame_in, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            area_proportion = area / self.info_input.area
            metrics[f"mask.{i}.area_proportion"] = area_proportion

            contour_bbox = cv2.boundingRect(contour)
            x, y, w, h = contour_bbox
            x_prop = x / self.info_input.x
            y_prop = y / self.info_input.y
            ratio = h / w
            metrics[f"mask.{i}.x_prop"] = x_prop
            metrics[f"mask.{i}.y_prop"] = y_prop
            metrics[f"mask.{i}.ratio"] = ratio
            # TODO identify same contour frame-to-frame for comparable stats
            break
        metrics.update(self.supplementary)
        self.queue_results.put((self.frame_no, metrics))

        np.copyto(frame_out, frame_in)
        return True

    def close(self) -> None:
        super().close()
        self.queue_results.put(None)


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
        super().__init__(info_input=info_input, queue_manager=queue_manager)
        self.res = (x, y)
        if fps_out > fps_in:
            msg = "fps_out cannot be greater than fps_in"
            raise ValueError(msg)
        self.fps_ratio = fps_in / fps_out

        # override default input-based output information from parent class
        self.info_output = FrameQueueInfoOutput(self.queue_resources.queue, x, y, info_input.colourspace)

    def process_frame(self, frame_in, frame_out) -> bool:
        if self.frame_no % self.fps_ratio < 1.0:
            cv2.resize(frame_in, self.res, frame_out)
            return True
        else:
            # skip frame
            return False
