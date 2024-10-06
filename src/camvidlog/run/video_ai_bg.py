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
            rescaler_down = Rescaler(
                info_input=file_reader.info_output,
                queue_manager=q_manager,
                x=1024,
                y=1024,
                fps_in=30,
                fps_out=5,
            )

            bgrem = BiRefNet(info_input=rescaler_down.info_output, queue_manager=q_manager)

            rescaler_2 = Rescaler(
                info_input=bgrem.info_output,
                queue_manager=q_manager,
                x=384,
                y=384,
                fps_in=5,
                fps_out=5,
            )

            # TODO crop (part of OpenClip)
            # identify
            mammalia = "Eukaryota Animalia Chordata Mammalia"
            queries = [
                mammalia + " Artiodactyla Cervidae Cervinae Muntiacini Muntiacus (muntjac deer)",
                mammalia + " Carnivora Feliformia Felidae Felinae Felis Felis catus (domestic cat)",
                mammalia + " Eulipotyphla Erinaceidae Erinaceus Erinaceus europaeus (European hedgehog)",
                mammalia + " Carnivora Canidae Vulpes Vulpes vulpes (red fox)",
                mammalia + " Carnivora Mustelidae Mustela Mustela furo (domestic ferret)",
            ]
            ai_clip = OpenClip(
                info_input=rescaler_2.info_output,
                queries=queries,
                data_recorder=data_recorder,
                supplementary={"bgrem": "yes"},
            )
            ps = []
            ps.append(Process(target=file_reader))
            ps.append(Process(target=rescaler_down))
            ps.append(Process(target=bgrem))
            ps.append(Process(target=rescaler_2))
            ps.append(Process(target=ai_clip))
            ps.append(Process(target=data_recorder))

            starttime = time.time()

            for p in ps:
                p.start()

            for p in ps:
                p.join()

            endtime = time.time()
