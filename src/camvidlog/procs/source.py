import argparse
from contextlib import contextmanager
from multiprocessing import JoinableQueue, Process
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path

import cv2
import numpy as np


@contextmanager
def video_capture_context(videopath: str):
    video_capture = cv2.VideoCapture(videopath, cv2.CAP_ANY)
    try:
        yield video_capture
    finally:
        video_capture.release()


def _vid_reader(
    videopath: Path,
    shared_memory_name: str,
    output_queue: JoinableQueue,
):
    with video_capture_context(str(videopath)) as video_capture:
        fps = float(video_capture.get(cv2.CAP_PROP_FPS))
        x = cv2.CAP_PROP_FRAME_WIDTH
        y = cv2.CAP_PROP_FRAME_HEIGHT

        frame_no = 0
        frame_time = 0.0
        shared_memory = None
        shared_array = None
        sucess = True
        while sucess:
            # first frame is read without shared memory
            sucess, frame = video_capture.read(shared_array)
            if sucess:
                if frame is not None and shared_memory is None:
                    # create the shared memory _after_ reading the first frame
                    # so we know how much shared memory we need
                    shared_memory = SharedMemory(name=shared_memory_name, create=True, size=frame.nbytes)
                    shared_array = np.ndarray(frame.shape, dtype=frame.dtype, buffer=shared_memory.buf)
                    shared_array[:] = frame[:]
                output_queue.put((frame_no, frame_time, frame.shape, frame.dtype))
                # wait for the queue to be read from
                output_queue.join()
                # move on to next frame
                frame_no += 1
                frame_time = frame_no / fps

        # put the sentinel into the queue
        output_queue.put(None)
        # wait for the sentinel to be read
        # otherwise shared memory of the last frame will close too soon
        output_queue.join()
        shared_memory.close()
        shared_memory.unlink()


def _vid_saver(
    shared_memory_name: str,
    output_queue: JoinableQueue,
):
    shared_memory = None
    shared_array = None
    # get things in the queue until the sentinel
    while item := output_queue.get():
        frame_no, frame_time, shape, dtype = item
        if not shared_memory:
            shared_memory = SharedMemory(name=shared_memory_name, create=False)
            shared_array = np.ndarray(shape, dtype=dtype, buffer=shared_memory.buf)
        print(frame_no)
        print(shared_memory.buf)
        print(shared_array.sum())
        output_queue.task_done()

    shared_memory.close()
    # say we've processed the sentinel
    output_queue.task_done()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    args = parser.parse_args()

    queue_reader = JoinableQueue(1)

    p_reader = Process(
        target=_vid_reader,
        args=(Path(args.filename), "frame", queue_reader),
    )
    p_printer = Process(target=_vid_saver, args=("frame", queue_reader))

    p_reader.start()
    p_printer.start()

    p_printer.join()
    p_reader.join()
