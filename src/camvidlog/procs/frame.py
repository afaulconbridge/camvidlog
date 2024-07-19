from multiprocessing import Queue

import cv2
import numpy as np

from camvidlog.procs.basics import Colourspace, FrameConsumer, FrameConsumerProducer, FrameQueueInfoOutput


class SaveToFile(FrameConsumer):
    filename: str
    fps: float
    out: cv2.VideoWriter | None

    def __init__(self, filename: str, fps: float, info_input: FrameQueueInfoOutput):
        super().__init__(info_input)
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

    def cleanup(self) -> None:
        if self.out:
            self.out.release()


class BackgroundSubtractorMOG2(FrameConsumerProducer):
    background_subtractor: cv2.BackgroundSubtractorMOG2

    def __init__(
        self,
        info_input: FrameQueueInfoOutput,
        queue_out: Queue,
        shared_memory_names_out: tuple[str, ...],
        history: int = 500,
        var_threshold=16,
    ):
        super().__init__(
            info_input=info_input,
            queue_out=queue_out,
            shared_memory_names_out=shared_memory_names_out,
            x=info_input.x,
            y=info_input.y,
            colourspace=info_input.colourspace,
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
        info_input: FrameQueueInfoOutput,
        queue_out: Queue,
        shared_memory_names_out: tuple[str, ...],
        history: int = 500,
        dist2_threshold: float = 400.0,
    ):
        super().__init__(
            info_input=info_input,
            queue_out=queue_out,
            shared_memory_names_out=shared_memory_names_out,
            x=info_input.x,
            y=info_input.y,
            colourspace=Colourspace.greyscale,
        )
        self.background_subtractor = cv2.createBackgroundSubtractorKNN(
            history=history, detectShadows=False, dist2Threshold=dist2_threshold
        )

    def process_frame(self, frame_in, frame_out) -> bool:
        fgmask = self.background_subtractor.apply(frame_in)
        cv2.cvtColor(fgmask, cv2.COLOR_GRAY2RGB, frame_out)
        return True


class BackgroundMaskDenoiser(FrameConsumerProducer):
    kernel_size: int
    frame_temp: None | np.ndarray = None

    def __init__(
        self,
        info_input: FrameQueueInfoOutput,
        queue_out: Queue,
        shared_memory_names_out: tuple[str, ...],
        kernel_size: int = 3,
    ):
        super().__init__(
            info_input=info_input,
            queue_out=queue_out,
            shared_memory_names_out=shared_memory_names_out,
            x=info_input.x,
            y=info_input.y,
            colourspace=info_input.colourspace,
        )
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


class Rescaler(FrameConsumerProducer):
    res: tuple[int, int]
    fps_ratio: float

    def __init__(
        self,
        info_input: FrameQueueInfoOutput,
        queue_out: Queue,
        shared_memory_names_out: tuple[str, ...],
        x: int,
        y: int,
        fps_in: int,
        fps_out: int,
    ):
        super().__init__(
            info_input=info_input,
            queue_out=queue_out,
            shared_memory_names_out=shared_memory_names_out,
            x=x,
            y=y,
            colourspace=info_input.colourspace,
        )
        self.res = (x, y)
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
