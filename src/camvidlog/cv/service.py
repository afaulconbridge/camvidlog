from pathlib import Path

from camvidlog.cv.models import ThingResult


class ComputerVisionService:
    def __init__(self):
        raise NotImplementedError()

    def find_things(self, videopath: Path | str, *, framestep=10) -> tuple[ThingResult, ...]:
        raise NotImplementedError()
