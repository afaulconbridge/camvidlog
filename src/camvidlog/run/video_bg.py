import argparse
import time
from multiprocessing import Process, Queue

from camvidlog.procs.basics import (
    FFMPEGReader,
    peek_in_file,
)
from camvidlog.procs.bg import BackgroundSubtractorMOG2
from camvidlog.procs.frame import (
    FFMPEGToFile,
    Rescaler,
)
from camvidlog.queues import SharedMemoryQueueManager

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", nargs="+")
    args = parser.parse_args()

    for filename in args.filename:
        vidstats = peek_in_file(filename)

        # need to assign shared memory from the parent process
        # otherwise it will be eagerly cleaned up when the child terminates
        q_manager = SharedMemoryQueueManager()
        q_results = Queue()
        ps = []
        with q_manager:
            # file_reader = FileReader(queue_manager=q_manager, filename=filename)
            file_reader = FFMPEGReader(queue_manager=q_manager, filename=filename)
            rescaler = Rescaler(
                info_input=file_reader.info_output,
                queue_manager=q_manager,
                x=vidstats.x // 4,
                y=vidstats.y // 4,
                fps_in=30,
                fps_out=5,
            )

            background_subtractor = BackgroundSubtractorMOG2(
                info_input=rescaler.info_output,
                queue_manager=q_manager,
                history=900,
                var_threshold=5,
                output_image_filename=f"{filename}.bg.mog2.jpg",
            )
            save_to_file = FFMPEGToFile(f"{filename}.mog2.mp4", 5, background_subtractor.info_output_bg)
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
