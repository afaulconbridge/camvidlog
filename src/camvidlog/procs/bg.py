import logging
from multiprocessing import Queue

import cv2
import numpy as np

from camvidlog.frameinfo import Colourspace, FrameQueueInfoOutput
from camvidlog.procs.basics import DataRecorder, FrameConsumerProducer
from camvidlog.queues import SharedMemoryQueueManager

logger = logging.getLogger(__name__)


class BackgroundSubtractorMOG2(FrameConsumerProducer):
    _background_subtractor: cv2.BackgroundSubtractorMOG2 | None = None
    history: int
    var_threshold: int
    output_image_filename = str

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

    def close(self) -> None:
        super().close()


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
        data_recorder: DataRecorder,
        prefix: str = "",
        supplementary: dict[str, str] | None = None,
    ):
        super().__init__(info_input=info_input, queue_manager=queue_manager)
        self.prefix = f"{prefix}." if prefix else ""
        self.supplementary = supplementary if supplementary else {}
        columns = (
            "mask.mean",
            "mask.0.area_proportion",
            "mask.0.x_prop",
            "mask.0.y_prop",
            "mask.0.ratio",
            *self.supplementary.keys(),
        )
        self.queue_results = data_recorder.register(columns)

    def process_frame(self, frame_in, frame_out) -> bool:
        metrics = {}
        mean = frame_in.mean() / 255
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
