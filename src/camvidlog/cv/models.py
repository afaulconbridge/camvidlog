from typing import List

from numpy import ndarray
from pydantic import BaseModel, ConfigDict


class ThingResultFrame(BaseModel):
    # needed to support numpy arrays
    model_config = ConfigDict(arbitrary_types_allowed=True)

    x1: int
    x2: int
    y1: int
    y2: int
    sub_img: ndarray


class ThingResult(BaseModel):
    frame_first: int
    frame_last: int
    frames: List[ThingResultFrame]
    result: str
    result_score: float
