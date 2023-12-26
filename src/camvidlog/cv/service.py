import logging
from typing import Generator

from ultralytics import YOLO
from ultralytics.engine.results import Results

from camvidlog import get_data
from camvidlog.cv.models import ThingResult, ThingResultFrame

logger = logging.getLogger(__name__)

"""

two step process

1. find - each tracked object is sequence of boxes from frames

2. know - extract boxes from frame and classify, then combine classification over track

"""


class ComputerVisionService:
    model_track: YOLO

    def __init__(self):
        self.model_track = YOLO(get_data("yolov8x-oiv7.pt"))
        # image net does not have good wildlife classes
        # openimage has some at least
        self.model_cls = YOLO(get_data("yolov8x-cls.pt"))

    def find_things(self, videopath, *, framestep=10, verbose=False) -> tuple[ThingResult, ...]:
        tracks = {}

        # iou = "intersection over union" for non-maximum supression (NMS)
        # tl;dr remove overlapping hits, lower is stricter
        results: Generator[Results, None, None] = self.model_track.track(
            videopath, stream=True, iou=0.25, agnostic_nms=True, vid_stride=framestep, verbose=verbose
        )
        for i, result in enumerate(results):
            logger.info(f"Processing frame {i*framestep}")

            # has to be explicit comparison to None
            # tensor may be falsy if its an id of 0
            # tensor throws error if more than one value
            if result.boxes.id is None:
                continue

            for box in result.cpu().boxes:
                box_id = box.id.item()

                # extract that region from the original image
                # TODO enlarge region 10% on each side first
                x1 = int(box.xyxy[0, 0])
                y1 = int(box.xyxy[0, 1])
                x2 = int(box.xyxy[0, 2])
                y2 = int(box.xyxy[0, 3])
                sub_img = result.orig_img[y1:y2, x1:x2]

                if not (thingresult := tracks.get(box_id)):
                    thingresult = ThingResult(frame_first=i, frame_last=i, frames=[])

                thingresult.frame_last = i
                thingresult.frames.append(ThingResultFrame(x1=x1, x2=x2, y1=y1, y2=y2, sub_img=sub_img))

                tracks[box_id] = thingresult

        return tuple(tracks.values())

    def know_tracks(self, tracks, *, verbose=False):
        knowns = {}
        for id_, track in tracks.items():
            probs = None
            names = None
            for i, xyxy, img in track:
                results = self.model_cls(img, verbose=verbose)
                if probs is not None:
                    probs += results[0].probs.cpu().numpy().data
                else:
                    probs = results[0].probs.cpu().numpy().data

                if not names:
                    names = results[0].names

            # now get the most prob and the name of it
            probs_sorted = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)
            knowns[id_] = tuple((p / len(track), names[i]) for i, p in probs_sorted[:5])
        return knowns
