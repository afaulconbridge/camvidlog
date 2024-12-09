import logging
import time
from collections.abc import Callable
from multiprocessing import Process

logger = logging.getLogger(__name__)


class ProcessManager:
    processes: list[Process]

    def __init__(self):
        self.processes = []

    def add(self, target: Callable[(...), None], name: str | None = None):
        if name is None:
            name = repr(target)
        self.processes.append(Process(target=target, name=name))
        return target

    def run_all(self):
        starttime = time.time()

        for p in self.processes:
            p.start()

        stop = False
        while not stop and any(p.exitcode is None for p in self.processes):
            for p in self.processes:
                if p.exitcode is None:
                    logger.info(f"Process {p.name} is alive")
                elif p.exitcode == 0:
                    logger.info(f"Process {p.name} terminated")
                elif p.exitcode != 0:
                    logger.info(f"Process {p.name} exited with {p.exitcode}")
                    stop = True
            time.sleep(1.0)

        for p in self.processes:
            if p.exitcode is None:
                p.terminate()
                p.kill()
            p.join(0)
            p.close()

        endtime = time.time()

        logger.info(f"Runtime: {endtime-starttime:0.2f}")
