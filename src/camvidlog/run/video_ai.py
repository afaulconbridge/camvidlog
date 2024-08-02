import argparse
import time
from collections.abc import Iterable
from multiprocessing import Process, Queue

from camvidlog.procs.ai import GroundingDino
from camvidlog.procs.basics import (
    DataRecorder,
    FileReader,
    Resolution,
    SharedMemoryQueueResources,
    peek_in_file,
)
from camvidlog.procs.frame import Rescaler
from camvidlog.procs.queues import SharedMemoryQueueManager

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", nargs="+")
    args = parser.parse_args()
    filenames: tuple[str] = tuple(args.filename)

    for filename in filenames:
        vidstats = peek_in_file(filename)

        # need to assign shared memory from the parent process
        # otherwise it will be eagerly cleaned up when the child terminates
        q_manager = SharedMemoryQueueManager()

        q2 = SharedMemoryQueueResources(vidstats.nbytes)
        q_results = Queue()

        file_reader = FileReader(q_manager, filename)
        rescaler = Rescaler(
            info_input=file_reader.info_output,
            queue_out=q2.queue,
            shared_memory_names_out=q2.shared_memory_names,
            x=Resolution.UHD.value[0],
            y=Resolution.UHD.value[1],
            fps_in=30,
            fps_out=1,
        )
        ai_grounding_dino = GroundingDino(
            info_input=rescaler.info_output,
            queries=["animal"],
            queue_results=q_results,
            model_id="IDEA-Research/grounding-dino-base",
            box_threshold=0.1,
            text_threshold=0.1,
        )
        data_recorder = DataRecorder(q_results, 1, filename.replace(".MP4", ".csv"))

        with q2:
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
