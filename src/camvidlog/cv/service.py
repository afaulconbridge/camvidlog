import logging
from pathlib import Path
from typing import Generator

import cv2
import numpy as np
import torch
from transformers import OwlViTForObjectDetection, OwlViTProcessor

from camvidlog.config import ConfigService
from camvidlog.cv.models import ThingResult, ThingResultFrame

logger = logging.getLogger(__name__)

# from https://stackoverflow.com/a/73261016/932342
Mat = np._typing.NDArray[np.uint8]

"""
two step process

1. find - each tracked object is sequence of boxes from frames

2. know - extract boxes from frame and classify, then combine classification over track
"""


def generate_frames(videopath: Path | str) -> Generator[Mat, None, None]:
    capture = cv2.VideoCapture(str(videopath), cv2.CAP_ANY)
    try:
        # get first frame
        sucess, frame = capture.read()
        yield frame

        while sucess:
            # handle frame
            yield frame

            # get next frame
            sucess, frame = capture.read()
    finally:
        capture.release()


def generate_frames_in_step(videopath: Path | str, framestep: int) -> Generator[Mat, None, None]:
    for i, frame in enumerate(generate_frames(videopath)):
        if i % framestep:
            continue
        yield frame


# Malisiewicz et al.
def non_max_suppression(boxes: np.ndarray, overlap_threshold: float = 0.3) -> tuple[int]:
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # NB: bounding boxes should be floats

    # initialize the list of picked indexes
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlap_threshold)[0])))

    # return the index of the bounding boxes that were picked
    return tuple(pick)


class ComputerVisionService:
    def __init__(self):
        # Large is really large, too big for 12GB
        # self.processor = OwlViTProcessor.from_pretrained("google/owlvit-large-patch14")
        # self.model = OwlViTForObjectDetection.from_pretrained("google/owlvit-large-patch14")
        # self.processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch16")
        # self.model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch16")
        self.processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        self.model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

        # OWLv2 puts boxes in weird places
        # Large is really large, too big for 12GB
        # self.processor = Owlv2Processor.from_pretrained("google/owlv2-large-patch14")
        # self.model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-large-patch14")
        # self.processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16")
        # self.model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16")

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)

    def find_things(self, videopath: Path | str, *, framestep=10) -> tuple[ThingResult, ...]:
        things: list[ThingResult] = []

        for step_i, frame in enumerate(generate_frames_in_step(videopath, framestep)):
            frame_i = step_i * framestep

            # TODO use a better list of potential matches
            potential_matches = [
                "deer",
                "fox",
                "hedgehog",
                "otter",
                "human",
                "cat",
                "dog",
                "mouse",
                "rat",
                "ferret",
                "badger",
                "bird",
            ]
            # 1 # texts = [[f"greyscale {text}" for text in texts[0]]]
            # 1 # overlap_threshold = 0.8  # go high to avoid false positives
            # 1 # confidence_threshold = 0.2  # go low and recheck later after tracking
            # 2 # texts = [[f"photo of {text}" for text in texts[0]]]
            # 3 # texts = [[f"black and white photo of {text}" for text in texts[0]]]
            # 4 # texts = [[f"black and white photo of {text}" for text in texts[0]]]
            # 4 # overlap_threshold = 0.6  # go high to avoid false positives
            # 4 # confidence_threshold = 0.3  # go low and recheck later after tracking
            # 4 # [results good, 0041-0901 (861) no mismatch]
            # 5 # (without contrast correction)
            texts = [[f"black and white photo of {text}" for text in potential_matches]]
            inputs = self.processor(text=texts, images=frame, return_tensors="pt")
            inputs.to(self.device)
            outputs = self.model(**inputs)

            # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
            target_sizes = torch.Tensor([[frame.shape[0], frame.shape[1]]])
            # Convert outputs (bounding boxes and class logits) to COCO API
            results = self.processor.post_process_object_detection(
                outputs=outputs, threshold=ConfigService.cv_threshold_detection, target_sizes=target_sizes
            )

            # Retrieve predictions for the first image for the corresponding text queries
            # pull everthing back to CPU
            boxes = results[0]["boxes"].cpu().detach().numpy()
            scores = results[0]["scores"].cpu().detach().numpy()
            labels = results[0]["labels"].cpu().detach().numpy()
            # cleanup GPU resources
            del inputs
            del results

            # remvove duplicates if more than one hit
            if boxes.shape[0] > 1:
                logger.debug(f"got {boxes.size} hits")
                box_keep = non_max_suppression(boxes, ConfigService.cv_threshold_tracking)
                # only keep the important result(s)
                boxes = boxes[box_keep,]
                scores = scores[box_keep,]
                labels = labels[box_keep,]

            # Print detected objects and rescaled box coordinates
            for j, (box, score, label) in enumerate(zip(boxes, scores, labels)):
                x1, y1, x2, y2 = (int(i) for i in box.tolist())
                w = x2 - x1
                h = y2 - y1
                result_score = score.item()  # unwrap the ndarray
                result = potential_matches[label]
                logger.debug(
                    f"frame {frame_i} hit {j} {result} with confidence {result_score:.2f} at location [{x1},{y1},{x2},{y2}] ({w}x{h})"
                )

                # TODO check if it overlaps with the last frame of a previous thing
                box_area = w * h
                result_thing = None
                for thing in reversed(things):
                    # check this was a thing recently found
                    # if thing.frame_last < i - framestep*trackinghistory:
                    #     logger.info(f"Skipping thing last seen in {thing.frame_last}")
                    #     continue

                    tx1 = thing.frames[-1].x1
                    ty1 = thing.frames[-1].y1
                    tx2 = thing.frames[-1].x2
                    ty2 = thing.frames[-1].y2

                    xx1 = max(x1, tx1)
                    yy1 = max(y1, ty1)
                    xx2 = min(x2, tx2)
                    yy2 = min(y2, ty2)

                    overlap_area = (xx2 - xx1) * (yy2 - yy1)
                    ratio = overlap_area / box_area

                    if ratio > ConfigService.cv_threshold_tracking:
                        # overlap!
                        result_thing = thing
                        logger.debug(f"Found existing thing at [{tx1},{ty1},{tx2},{ty2}] ({ratio:.2f}%)")
                        break

                if not result_thing:
                    # no existing thing found, make a new one
                    result_thing = ThingResult(
                        frame_first=frame_i,
                        frame_last=frame_i,
                        frames=[],
                        result=potential_matches[label],
                        result_score=result_score,
                    )
                    things.append(result_thing)
                    logger.debug("Created new thing")

                # extract that region from the original image
                # enlarge region 10% on each side
                ew = w // 10
                eh = h // 10
                sub_img = frame[y1 - eh : y2 + eh, x1 - ew : x2 + ew]

                # extend thing with this new hit
                # TODO fill in any missing frames images
                result_thing.frame_last = frame_i
                result_thing.frames.append(
                    ThingResultFrame(
                        x1=x1,
                        x2=x2,
                        y1=y1,
                        y2=y2,
                        sub_img=sub_img,
                        result=result,
                        result_score=result_score,
                    )
                )

        # filter to only things that appeared for a minimum time
        return [thing for thing in things if len(thing.frames) > (30 / framestep) * ConfigService.cv_time_min_tracking]

    def know_tracks(self, tracks, *, verbose=False):
        raise NotImplementedError()
