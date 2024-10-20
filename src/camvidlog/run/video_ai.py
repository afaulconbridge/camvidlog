import argparse
import logging
from multiprocessing import Queue

from camvidlog.procs.ai import OpenClip
from camvidlog.procs.basics import (
    DataRecorder,
    FFMPEGReader,
    FrameCopier,
    peek_in_file,
)
from camvidlog.procs.frame import Rescaler
from camvidlog.procs.manager import ProcessManager
from camvidlog.queues import SharedMemoryQueueManager

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", nargs="+")
    args = parser.parse_args()
    filenames: tuple[str] = tuple(args.filename)

    for filename in filenames:
        vidstats = peek_in_file(filename)
        pman = ProcessManager()

        # need to assign shared memory from the parent process
        # otherwise it will be eagerly cleaned up when the child terminates
        q_manager = SharedMemoryQueueManager()
        with q_manager:
            data_recorder = DataRecorder(Queue(), 1, filename + ".ai.csv")
            pman.add(target=data_recorder, name="DataRecorder")

            # file_reader = FileReader(queue_manager=q_manager, filename=filename)
            file_reader = FFMPEGReader(queue_manager=q_manager, filename=filename)
            pman.add(target=file_reader, name="Reader")

            copier = FrameCopier(file_reader.info_output, q_manager, 4)
            pman.add(target=copier, name="Copier")

            for i, (x, y) in enumerate(
                (
                    (vidstats.x, vidstats.y),
                    (vidstats.x // 2, vidstats.y // 2),
                    (vidstats.x // 4, vidstats.y // 4),
                    (384, 384),
                )
            ):
                rescaler = Rescaler(
                    info_input=copier.info_outputs[i],
                    # info_input=file_reader.info_output,
                    queue_manager=q_manager,
                    x=x,
                    y=y,
                    fps_in=30,
                    fps_out=5,
                )
                pman.add(target=rescaler, name=f"Rescaler {x}x{y}")

                queries = [
                    "deer",
                    "cat",
                    "hedgehog",
                    "fox",
                    "otter",
                    "mink",
                    "badger",
                    "ferret",
                    "rat",
                    "mouse",
                    "mole",
                ]
                queries = [
                    "Eukaryota Animalia Chordata Mammalia Artiodactyla Cervidae Cervinae Muntiacini Muntiacus (muntjac deer)",
                    "Eukaryota Animalia Chordata Mammalia Carnivora Feliformia Felidae Felinae Felis Felis catus (domestic cat)",
                    "Eukaryota Animalia Chordata Mammalia Eulipotyphla Erinaceidae Erinaceus Erinaceus europaeus (European hedgehog)",
                    "Eukaryota Animalia Chordata Mammalia Carnivora Canidae Vulpes Vulpes vulpes (red fox)",
                    # "otter",
                    # "mink",
                    # "badger",
                    "Eukaryota Animalia Chordata Mammalia Carnivora Mustelidae Mustela Mustela furo (domestic ferret)",
                    # "rat",
                    # "mouse",
                    # "mole",
                    # "not deer and not cat and not hedgehog and not fox and not ferret",
                ]

                ai_clip = OpenClip(
                    info_input=rescaler.info_output,
                    queries=queries,
                    data_recorder=data_recorder,
                    supplementary={"res": f"{x}x{y}"},
                )
                pman.add(target=ai_clip, name=f"OpenClip {x}x{y}")

            pman.run_all()
