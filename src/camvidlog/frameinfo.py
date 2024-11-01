from dataclasses import dataclass
from enum import Enum
from multiprocessing import JoinableQueue

import numpy as np


class Resolution(Enum):
    # Y,X to match openCV
    VGA = (480, 854)  # FWVGA
    SD = (720, 1280)
    HD = (1080, 1920)  # 2k, full HD
    UHD = (2160, 3840)  # 4K, UHD


class Colourspace(Enum):
    RGB = "rgb"
    greyscale = "greyscale"
    mask = "mask"


@dataclass
class FrameQueueInfoOutput:
    queue: JoinableQueue
    x: int
    y: int
    colourspace: Colourspace

    @property
    def shape(self) -> tuple[int, int, int]:
        # Y,X to match openCV
        return (self.y, self.x, 1 if self.colourspace == Colourspace.greyscale else 3)

    @property
    def area(self) -> int:
        return self.x * self.y

    @property
    def nbytes(self) -> int:
        return (
            self.x * self.y * (1 if self.colourspace == Colourspace.greyscale else 3) * (np.iinfo(np.uint8).bits // 8)
        )


@dataclass
class VideoFileStats:
    filename: str
    fps: float
    x: int
    y: int
    colourspace: Colourspace

    @property
    def shape(self) -> tuple[int, int, int]:
        # Y,X to match openCV
        return (self.y, self.x, 1 if self.colourspace == Colourspace.greyscale else 3)

    @property
    def nbytes(self) -> int:
        if self.colourspace == Colourspace.greyscale:
            return self.x * self.y
        else:
            return self.x * self.y * 3
