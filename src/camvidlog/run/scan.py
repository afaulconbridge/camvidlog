import argparse
import json
import logging
import os
from pathlib import Path
from typing import Generator, Iterable, Optional

import cv2
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, GroundingDinoProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class VideoManager:
    videopath: Path | str
    fps: Optional[float] = None
    capture: Optional[cv2.VideoCapture]

    def __init__(self, videopath: Path | str):
        self.videopath = videopath

    def __enter__(self):
        self.capture = cv2.VideoCapture(str(self.videopath), cv2.CAP_ANY)
        return self

    def __exit__(self, type, value, traceback):
        self.capture.release()
        self.capture = None
        self.fps = None

    def generate_frames(self) -> Generator[tuple[int, float, cv2.typing.MatLike], None, None]:
        frame_no = 0
        frame_time = 0.0  # time since start in seconds
        # get first frame
        sucess, frame = self.capture.read()
        # TODO verify RGB?

        if not self.fps:
            self.fps = float(self.capture.get(cv2.CAP_PROP_FPS))

        while sucess:
            # handle frame
            yield frame_no, frame_time, frame

            # get next frame
            sucess, frame = self.capture.read()
            frame_no += 1
            frame_time = frame_no / self.fps


class ImageManager:
    model_id: str
    text_queries: str
    threshold_box: float
    threshold_text: float

    def __init__(
        self,
        queries: Iterable[str],
        model_id="IDEA-Research/grounding-dino-base",
        threshold_box: float = 0.25,
        threshold_text: float = 0.25,
    ):
        self.model_id = model_id
        self.text_queries = ". ".join(queries) + "."
        self.threshold_box = threshold_box
        self.threshold_text = threshold_text

    def __enter__(self):
        logger.debug("Creating Autoprocessor")
        self.processor: GroundingDinoProcessor = AutoProcessor.from_pretrained(self.model_id)
        # process the text input now, as it doesn't change frame-to-frame
        self.processor_input_ids = self.processor(text=self.text_queries, return_tensors="pt").to("cuda")
        logger.debug("Creating AutoModelForZeroShotObjectDetection")
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id).to("cuda")
        return self

    def __exit__(self, type, value, traceback):
        pass

    def process(self, image: cv2.typing.MatLike) -> Generator[tuple[float, str, tuple[int, int, int, int]], None, None]:
        image_pillow = Image.fromarray(image)
        inputs = self.processor(images=image_pillow, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = self.model(**inputs, **self.processor_input_ids)
            results = self.processor.post_process_grounded_object_detection(
                outputs,
                self.processor_input_ids.input_ids,
                box_threshold=self.threshold_box,
                text_threshold=self.threshold_text,
                target_sizes=[image_pillow.size[::-1]],
            )[0]
        for score, label, bbox in zip(*results.values()):
            yield float(score), str(label), [int(i) for i in bbox]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="CamVidLogScan", description="Scans video files to find things")
    parser.add_argument("--dryrun", "-d", action="store_true")
    parser.add_argument("filename", nargs="+")
    args = parser.parse_args()

    # hedgehog - very hard for ML!
    # data/20231119173624_VD_00333.MP4

    queries = "person", "cat", "deer", "bird", "otter", "fox", "hedgehog", "mouse", "rat"

    for filename in args.filename:
        output = str(Path(filename).with_suffix("").with_suffix(".json"))

        # merge with existing file if it exists
        if os.path.exists(output):
            with open(output) as outfile:
                results = json.load(outfile)
                result_frames = results["frames"]
        else:
            results = {}
            result_frames = []
            results["frames"] = result_frames

        with ImageManager(queries) as imagemanager:
            with VideoManager(filename) as videomanager:
                process_time = 0.0
                process_interval = 1.0 / 10
                for i, time, frame in videomanager.generate_frames():
                    if "fps" not in results:
                        results["fps"] = videomanager.fps

                    if time >= (process_time - 0.00000001):  # handle float point error
                        logger.debug(f"process {i} {time}")
                        for score, label, bbox in imagemanager.process(frame):
                            result_frame = {}
                            result_frame["number"] = i
                            result_frame["time"] = time
                            result_frame["score"] = score
                            result_frame["label"] = label
                            result_frame["bbox"] = bbox
                            result_frame["model"] = imagemanager.model_id
                            result_frame["query"] = imagemanager.text_queries
                            result_frames.append(result_frame)
                        # move to next slot
                        process_time += process_interval

        if not args.dryrun:
            with open(output, "w") as outfile:
                json.dump(results, outfile, indent=2, sort_keys=True)
