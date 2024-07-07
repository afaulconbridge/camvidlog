import argparse
from multiprocessing import Process, Queue
from multiprocessing.shared_memory import SharedMemory

import cv2
import numpy as np

from camvidlog.procs.basics import (
    FileReader,
    FrameConsumer,
    FrameConsumerProducer,
    SharedMemoryQueueResources,
    peek_in_file,
)


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
                cv2.VideoWriter_fourcc(*"MJPG"),
                # cv2.VideoWriter_fourcc(*"X264"),
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
        shape=tuple[int, int, int],
        dtype=np.dtype,
        history: int = 500,
        var_threshold=16,
    ):
        super().__init__(
            queue_in=queue_in,
            queue_out=queue_out,
            shared_memory_names_out=shared_memory_names_out,
            shape=shape,
            dtype=dtype,
        )
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history, detectShadows=False, varThreshold=var_threshold
        )

    def process_frame(self, frame_in, frame_out) -> bool:
        self.background_subtractor.apply(frame_in, frame_out)
        return True


class BackgroundSubtractorKNN(FrameConsumerProducer):
    background_subtractor: cv2.BackgroundSubtractorKNN

    def __init__(
        self,
        queue_in: Queue,
        queue_out: Queue,
        shared_memory_names_out: tuple[str, ...],
        shape=tuple[int, int, int],
        dtype=np.dtype,
        history: int = 500,
        dist2_threshold: float = 400.0,
    ):
        super().__init__(
            queue_in=queue_in,
            queue_out=queue_out,
            shared_memory_names_out=shared_memory_names_out,
            shape=shape,
            dtype=dtype,
        )
        self.background_subtractor = cv2.createBackgroundSubtractorKNN(
            history=history, detectShadows=False, dist2Threshold=dist2_threshold
        )

    def process_frame(self, frame_in, frame_out) -> bool:
        self.background_subtractor.apply(frame_in, frame_out)
        return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    args = parser.parse_args()

    vidstats = peek_in_file(args.filename)

    # need to assign shared memory from the parent process
    # otherwise it will be eagerly cleaned up when the child terminates
    q1 = SharedMemoryQueueResources(vidstats.nbytes)
    q2 = SharedMemoryQueueResources(vidstats.nbytes)

    fr = FileReader(
        args.filename,
        vidstats.fps,
        q1.queue,
        q1.shared_memory_names,
        vidstats.shape,
        vidstats.dtype,
    )

    bs = BackgroundSubtractorMOG2(q1.queue, q2.queue, q2.shared_memory_names, vidstats.shape, vidstats.dtype)
    stf = SaveToFile(
        "output.avi",
        vidstats.fps,
        q1.queue,
        vidstats.shape,
        vidstats.dtype,
    )

    with q1, q2:
        ps = []
        ps.append(Process(target=fr, args=()))
        # ps.append(Process(target=bs, args=()))
        ps.append(Process(target=stf, args=()))

        for p in ps:
            p.start()

        for p in ps:
            p.join()
