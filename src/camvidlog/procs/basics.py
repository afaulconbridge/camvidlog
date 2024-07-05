from contextlib import contextmanager
from multiprocessing import Queue
from multiprocessing.resource_tracker import unregister
from multiprocessing.shared_memory import SharedMemory

import cv2
import numpy as np

TIMEOUT = 60.0


@contextmanager
def video_capture_context(videopath: str):
    video_capture = cv2.VideoCapture(videopath, cv2.CAP_ANY)
    try:
        yield video_capture
    finally:
        video_capture.release()


class FileReader:
    videopath: str
    shared_memory_name: str

    def __init__(self, videopath: str, shared_memory_name: str):
        self.videopath = videopath
        self.shared_memory_name = shared_memory_name
        self.shared_memory = []
        self.shared_array = []

    def _init_memory(self, nbytes: int, shape, dtype):
        for i in range(2):
            name = f"{self.shared_memory_name}_{i}"
            shared_memory_obj = SharedMemory(name=name, create=True, size=nbytes)
            self.shared_memory.append(shared_memory_obj)
            self.shared_array.append(np.ndarray(shape, dtype=dtype, buffer=shared_memory_obj.buf))

            # dangerous - tells python that this is not the shared memory it is looking for
            # pro - avoids UserWarning because python is over-eager in cleaning ended proesses
            # con - could leak, fragile to internal implementation
            # unregister(shared_memory_obj._name, "shared_memory")

        self.shared_pointer = 0

    def __call__(self, output_queue: Queue):
        with video_capture_context(str(self.videopath)) as video_capture:
            fps = float(video_capture.get(cv2.CAP_PROP_FPS))
            x = cv2.CAP_PROP_FRAME_WIDTH
            y = cv2.CAP_PROP_FRAME_HEIGHT

            frame_no = 0
            frame_time = 0.0
            success = True
            while success:
                # first frame is read without shared memory
                if frame_no == 0:
                    success, frame = video_capture.read()
                    # create the shared memory _after_ reading the first frame
                    # so we know how much shared memory we need
                    self._init_memory(frame.nbytes, frame.shape, frame.dtype)
                    # put the image into shared memory
                    self.shared_array[self.shared_pointer][:] = frame[:]
                else:
                    # only work with two shared arrays
                    # any more and it hangs?!?
                    success, frame = video_capture.read(self.shared_array[self.shared_pointer])

                if success:
                    output_queue.put(
                        (self.shared_pointer, frame_no, frame_time, fps, frame.shape, frame.dtype), timeout=TIMEOUT
                    )
                    # move on to next frame
                    frame_no += 1
                    frame_time = frame_no / fps
                    if self.shared_pointer == 0:
                        self.shared_pointer = 1
                    else:
                        self.shared_pointer = 0

            for i in range(2):
                self.shared_memory[i].close()

            # put the sentinel into the queue
            output_queue.put(None)
            output_queue.close()


class FrameConsumer:
    def __init__(self, shared_memory_name: str):
        self.shared_memory_name = shared_memory_name

    def __call__(self, input_queue: Queue):
        shared_memory = []
        shared_array = []

        while item := input_queue.get(timeout=TIMEOUT):
            ring_pointer, frame_no, frame_time, fps, shape, dtype = item
            self.shape = shape
            self.fps = fps
            if ring_pointer >= len(shared_memory):
                shared_memory.append(SharedMemory(name=f"{self.shared_memory_name}_{ring_pointer}", create=False))
                shared_array.append(np.ndarray(shape, dtype=dtype, buffer=shared_memory[-1].buf))

            self.process_frame(shared_array[ring_pointer])

        self.cleanup()
        input_queue.close()

        for shared_memory_item in shared_memory:
            shared_memory_item.close()

    def process_frame(self, frame) -> None:
        pass

    def cleanup(self) -> None:
        pass


class FrameConsumerProducer:
    def __init__(self, shared_memory_in_name: str, shared_memory_out_name: str):
        self.shared_memory_in_name = shared_memory_in_name
        self.shared_memory_out_name = shared_memory_out_name

    def __call__(self, input_queue: Queue, output_queue: Queue):
        shared_memory_in = []
        shared_array_in = []
        shared_memory_out = []
        shared_array_out = []
        ring_pointer_out = 0

        while item := input_queue.get(timeout=TIMEOUT):
            ring_pointer_in, frame_no, frame_time, fps, shape, dtype = item

            # setup input frame
            if ring_pointer_in >= len(shared_memory_in):
                shared_memory_in.append(
                    SharedMemory(name=f"{self.shared_memory_in_name}_{ring_pointer_in}", create=False)
                )
                shared_array_in.append(np.ndarray(shape, dtype=dtype, buffer=shared_memory_in[-1].buf))

            # setup output
            if not shared_memory_out:
                for i in range(2):
                    shared_memory_obj = SharedMemory(
                        name=f"{self.shared_memory_out_name}_{i}",
                        create=True,
                        size=shared_array_in[ring_pointer_in].nbytes,
                    )
                    shared_memory_out.append(shared_memory_obj)
                    shared_array_out.append(np.ndarray(shape, dtype=dtype, buffer=shared_memory_obj.buf))

                    # dangerous - tells python that this is not the shared memory it is looking for
                    # pro - avoids UserWarning because python is over-eager in cleaning ended proesses
                    # con - could leak, fragile to internal implementation
                    # unregister(shared_memory_obj._name, "shared_memory")
                ring_pointer_out = 0

            # actually handle the processing
            self.process_frame(shared_array_in[ring_pointer_in], shared_array_out[ring_pointer_out])

            output_queue.put((ring_pointer_out, frame_no, frame_time, fps, shape, dtype), timeout=TIMEOUT)

            if ring_pointer_out == 0:
                ring_pointer_out = 1
            else:
                ring_pointer_out = 0

        input_queue.close()

        for shared_memory_item in shared_memory_in:
            shared_memory_item.close()

        # put the sentinel into the queue
        output_queue.put(None)
        output_queue.close()

    def process_frame(self, frame_in, frame_out) -> None:
        pass
