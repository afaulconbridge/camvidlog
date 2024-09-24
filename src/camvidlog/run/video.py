import argparse
import time
from multiprocessing import Process, Queue

from camvidlog.procs.basics import (
    DataRecorder,
    FFMPEGReader,
    peek_in_file,
)
from camvidlog.procs.frame import (
    BackgroundMaskDenoiser,
    BackgroundSubtractorMOG2,
    FFMPEGToFile,
    Rescaler,
)
from camvidlog.procs.queues import SharedMemoryQueueManager

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
                history=5000000,
                var_threshold=5,
            )
            background_mask_denoiser = BackgroundMaskDenoiser(
                info_input=background_subtractor.info_output, queue_manager=q_manager, kernel_size=5
            )
            save_to_file = FFMPEGToFile(f"{filename}.mog2.mp4", 5, background_mask_denoiser.info_output)
            ps.append(Process(target=file_reader))
            ps.append(Process(target=rescaler))
            ps.append(Process(target=background_subtractor))
            ps.append(Process(target=background_mask_denoiser))
            ps.append(Process(target=save_to_file))

            starttime = time.time()
            for p in ps:
                p.start()

            for p in ps:
                p.join()

            endtime = time.time()
