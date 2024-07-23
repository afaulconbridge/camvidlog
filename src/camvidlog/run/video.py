import argparse
import time
from multiprocessing import Process

from camvidlog.procs.basics import (
    FileReader,
    Resolution,
    peek_in_file,
)
from camvidlog.procs.frame import BackgroundMaskDenoiser, BackgroundSubtractorMOG2, Rescaler, SaveToFile
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
        with q_manager:
            file_reader = FileReader(queue_manager=q_manager, filename=filename)
            rescaler = Rescaler(
                info_input=file_reader.info_output,
                queue_manager=q_manager,
                x=Resolution.SD.value[1],
                y=Resolution.SD.value[0],
                fps_in=30,
                fps_out=5,
            )
            background_subtractor = BackgroundSubtractorMOG2(
                info_input=rescaler.info_output,
                queue_manager=q_manager,
                history=5000,
            )
            background_mask_denoiser = BackgroundMaskDenoiser(
                info_input=background_subtractor.info_output,
                queue_manager=q_manager,
            )
            save_to_file = SaveToFile("output.avi", 5, background_mask_denoiser.info_output)
            # mask_stats = MaskStats(background_mask_denoiser.info_output)

            ps = []
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
            print(f"Ran in {endtime-starttime:.2f}s")
