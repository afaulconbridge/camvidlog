from typing import Generator

from PIL import Image
from pydantic import BaseModel
from ultralytics import YOLO
from ultralytics.engine.results import Results

from camvidlog import get_data


class VideoResult(BaseModel):
    timestart: int
    timeend: int
    probs: tuple[tuple[int, float], ...]


def plot_result(result):
    im_array = result.plot()  # plot a BGR numpy array of predictions
    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    im.show()  # show image


class ComputerVisionService:
    model: YOLO

    def __init__(self):
        self.model = YOLO(get_data("yolov8x-oiv7.pt"))

    def analyse_video(self, videopath, framestep=3) -> tuple[VideoResult, ...]:
        tracks = {}
        # iou = "intersection over union" for non-maximum supression (NMS)
        # tl;dr remove overlapping hits, lower is stricter
        results: Generator[Results, None, None] = self.model.track(
            videopath, stream=True, iou=0.25, agnostic_nms=True, vid_stride=framestep
        )
        for i, result in enumerate(results):
            if not result.boxes.id:
                continue

            result.boxes.cpu()

            for box in result.boxes:
                box_id = box.id.item()
                box_conf = box.conf.item()
                box_cls = box.cls.item()
                print(f"{box_id} {box_cls} {box_conf}")
                if box_id in tracks:
                    previous_result = tracks[box_id]
                    tracks[box_id] = VideoResult(
                        timestart=previous_result.timestart,
                        timeend=i,
                        probs=(),
                    )
                elif i == 10:  # noqa: PLR2004
                    # tracking algorithm takes 10 images to get a good lock
                    # so assume its been there from the start if this is the 10th step
                    tracks[box_id] = VideoResult(timestart=0, timeend=i, probs=())
                else:
                    tracks[box_id] = VideoResult(timestart=i, timeend=i, probs=())

        return tuple(tracks.values())
