import argparse
import time
from multiprocessing import Process, Queue

from camvidlog.procs.ai import BiRefNet, OpenClip
from camvidlog.procs.basics import (
    DataRecorder,
    FFMPEGReader,
    peek_in_file,
)
from camvidlog.procs.frame import FFMPEGToFile, Rescaler
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
        with q_manager:
            data_recorder = DataRecorder(Queue(), 1, filename + ".ai.csv")

            # file_reader = FileReader(queue_manager=q_manager, filename=filename)
            file_reader = FFMPEGReader(queue_manager=q_manager, filename=filename)
            rescaler = Rescaler(
                info_input=file_reader.info_output,
                queue_manager=q_manager,
                x=vidstats.x,
                y=vidstats.y,
                fps_in=30,
                fps_out=1,
            )

            bgrem = BiRefNet(info_input=rescaler.info_output, queue_manager=q_manager)

            # TODO crop
            # TODO identify

            save_to_file = FFMPEGToFile(f"{filename}.bgrem.mp4", 1, bgrem.info_output)

            ps = []
            ps.append(Process(target=file_reader))
            ps.append(Process(target=rescaler))
            ps.append(Process(target=bgrem))
            ps.append(Process(target=save_to_file))

            starttime = time.time()

            for p in ps:
                p.start()

            for p in ps:
                p.join()

            endtime = time.time()
