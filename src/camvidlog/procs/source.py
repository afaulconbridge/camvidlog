import argparse
from multiprocessing import Process, Queue
from multiprocessing.shared_memory import SharedMemory

import cv2

from camvidlog.procs.basics import FileReader, FrameConsumer, FrameConsumerProducer


class SaveToFile(FrameConsumer):
    out: cv2.VideoWriter | None

    def __init__(self, shared_memory_name: str, filename):
        super().__init__(shared_memory_name=shared_memory_name)
        self.filename = filename
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
        super().cleanup()
        self.out.release()


class BackgroundSubtractorMOG2(FrameConsumerProducer):
    background_subtractor: cv2.BackgroundSubtractorMOG2

    def __init__(self, shared_memory_in_name: str, shared_memory_out_name: str, history: int = 500, var_threshold=16):
        super().__init__(shared_memory_in_name=shared_memory_in_name, shared_memory_out_name=shared_memory_out_name)
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history, detectShadows=False, varThreshold=var_threshold
        )

    def process_frame(self, frame_in, frame_out) -> None:
        self.background_subtractor.apply(frame_in, frame_out)


class BackgroundSubtractorKNN(FrameConsumerProducer):
    background_subtractor: cv2.BackgroundSubtractorKNN

    def __init__(
        self,
        shared_memory_in_name: str,
        shared_memory_out_name: str,
        history: int = 500,
        dist2_threshold: float = 400.0,
    ):
        super().__init__(shared_memory_in_name, shared_memory_out_name)
        self.background_subtractor = cv2.createBackgroundSubtractorKNN(
            history=history, detectShadows=False, dist2Threshold=dist2_threshold
        )

    def process_frame(self, frame_in, frame_out) -> None:
        self.background_subtractor.apply(frame_in, frame_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    args = parser.parse_args()

    ps = []
    processor_queue = Queue(1)
    ps.append(Process(target=FileReader(args.filename, "fa"), args=(processor_queue,)))
    sink_queue = Queue(1)
    ps.append(Process(target=BackgroundSubtractorMOG2("fa", "fb"), args=(processor_queue, sink_queue)))
    # ps.append(Process(target=BackgroundSubtractorKNN("fa", "fb"), args=(processor_queue, sink_queue)))
    ps.append(Process(target=SaveToFile("fb", "output.avi"), args=(sink_queue,)))

    processor_queue = Queue(1)
    # ps.append(Process(target=FileReader(args.filename, "fa"), args=(processor_queue,)))
    # ps.append(Process(target=SaveToFile("fa", "output.avi"), args=(processor_queue,)))

    for p in ps:
        p.start()

    for p in ps:
        p.join()

    for name in "fa", "fb":
        for i in (0, 1):
            shared_memory_obj = SharedMemory(name=f"{name}_{i}", create=False)
            shared_memory_obj.unlink()
