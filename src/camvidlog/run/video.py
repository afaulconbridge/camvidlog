import argparse
import time
from multiprocessing import Process, Queue

from camvidlog.procs.basics import (
    DataRecorder,
    FFMPEGReader,
    FileReader,
    FrameCopier,
    Resolution,
    peek_in_file,
)
from camvidlog.procs.frame import (
    BackgroundMaskDenoiser,
    BackgroundSubtractorMOG2,
    FFMPEGToFile,
    MaskStats,
    Rescaler,
    SaveToFile,
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
            file_reader = FileReader(queue_manager=q_manager, filename=filename)
            ps.append(Process(target=file_reader))

            # file_reader = FFMPEGReader(queue_manager=q_manager, filename=filename)
            copier = FrameCopier(file_reader.info_output, q_manager, len(Resolution))
            ps.append(Process(target=copier))
            for i, res in enumerate(Resolution):
                rescaler = Rescaler(
                    info_input=copier.info_outputs[i],
                    queue_manager=q_manager,
                    x=res.value[1],
                    y=res.value[0],
                    fps_in=30,
                    fps_out=5,
                )
                background_subtractor = BackgroundSubtractorMOG2(
                    info_input=rescaler.info_output, queue_manager=q_manager, history=500, var_threshold=16
                )
                background_mask_denoiser = BackgroundMaskDenoiser(
                    info_input=background_subtractor.info_output, queue_manager=q_manager, kernel_size=3
                )
                background_mask_stats = MaskStats(
                    info_input=background_mask_denoiser.info_output,
                    queue_manager=q_manager,
                    queue_results=q_results,
                    prefix=f"{res.value[1]}x{res.value[0]}",
                )
                save_to_file = FFMPEGToFile(
                    f"{filename}.{res.value[1]}x{res.value[0]}.mp4", 5, background_mask_stats.info_output
                )
                ps.append(Process(target=rescaler))
                ps.append(Process(target=background_subtractor))
                ps.append(Process(target=background_mask_denoiser))
                ps.append(Process(target=background_mask_stats))
                ps.append(Process(target=save_to_file))

            data_recorder = DataRecorder(q_results, len(Resolution), f"{filename}.csv")

            ps.append(Process(target=data_recorder))

            starttime = time.time()
            for p in ps:
                p.start()

            for p in ps:
                p.join()

            endtime = time.time()
            print(f"Ran in {endtime-starttime:.2f}s")
