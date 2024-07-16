import argparse
import time
from multiprocessing import Process, Queue
from typing import Iterable

from camvidlog.procs.ai import GroundingDino
from camvidlog.procs.basics import (
    DataRecorder,
    FileReader,
    Resolution,
    SharedMemoryQueueResources,
    peek_in_file,
)
from camvidlog.procs.frame import Rescaler

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", nargs="+")
    args = parser.parse_args()
    filenames: tuple[str] = tuple(args.filename)

    for filename in filenames:
        vidstats = peek_in_file(filename)

        # need to assign shared memory from the parent process
        # otherwise it will be eagerly cleaned up when the child terminates
        q1 = SharedMemoryQueueResources(vidstats.nbytes)
        q2 = SharedMemoryQueueResources(vidstats.nbytes)
        q_results = Queue()

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
            (*Resolution.UHD.value, 3),
            vidstats.dtype,
            vidstats.dtype,
            fps_in=30,
            fps_out=1,
        )
        ai_grounding_dino = GroundingDino(
            q2.queue,
            (*Resolution.UHD.value, 3),
            vidstats.dtype,
            ["animal"],
            q_results,
            "IDEA-Research/grounding-dino-base",
            box_threshold=0.1,
            text_threshold=0.1,
        )
        data_recorder = DataRecorder(q_results, 1, filename.replace(".MP4", ".csv"))

        with q1, q2:
            ps = []
            ps.append(Process(target=file_reader))
            ps.append(Process(target=rescaler))
            ps.append(Process(target=ai_grounding_dino))
            ps.append(Process(target=data_recorder))

            starttime = time.time()

            for p in ps:
                p.start()

            for p in ps:
                p.join()

            endtime = time.time()
            print(f"Ran in {endtime-starttime:.2f}s")

        print(data_recorder)
