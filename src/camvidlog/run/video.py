import argparse
import time
from multiprocessing import Process

from camvidlog.procs.basics import (
    FileReader,
    Resolution,
    SharedMemoryQueueResources,
    peek_in_file,
)
from camvidlog.procs.frame import BackgroundSubtractorMOG2, Rescaler, SaveToFile

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", nargs="+")
    args = parser.parse_args()

    for filename in args.filename:
        vidstats = peek_in_file(filename)

        # need to assign shared memory from the parent process
        # otherwise it will be eagerly cleaned up when the child terminates
        q1 = SharedMemoryQueueResources(vidstats.nbytes)
        q2 = SharedMemoryQueueResources(vidstats.nbytes)
        q3 = SharedMemoryQueueResources(vidstats.nbytes)

        file_reader = FileReader(
            filename,
            vidstats.fps,
            q1.queue,
            q1.shared_memory_names,
            vidstats.shape,
            vidstats.dtype,
        )
        rescaler = Rescaler(
            q1.queue,
            q2.queue,
            q2.shared_memory_names,
            vidstats.shape,
            (*Resolution.SD.value, 3),
            vidstats.dtype,
            vidstats.dtype,
            fps_in=30,
            fps_out=5,
        )
        background_subtractor = BackgroundSubtractorMOG2(
            q2.queue,
            q3.queue,
            q3.shared_memory_names,
            (*Resolution.SD.value, 3),
            vidstats.dtype,
        )
        save_to_file = SaveToFile(
            "output.avi",
            5,
            q3.queue,
            (*Resolution.SD.value, 3),
            vidstats.dtype,
        )

        with q1, q2, q3:
            ps = []
            ps.append(Process(target=file_reader))
            ps.append(Process(target=rescaler))
            ps.append(Process(target=background_subtractor))
            ps.append(Process(target=save_to_file))

            starttime = time.time()

            for p in ps:
                p.start()

            for p in ps:
                p.join()

            endtime = time.time()
            print(f"Ran in {endtime-starttime:.2f}s")
