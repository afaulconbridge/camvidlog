import logging
import time
from collections.abc import Generator, Iterable
from math import ceil, floor
from pathlib import Path

import cv2
import numpy as np
import torch
import typer
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

logger = logging.getLogger(__name__)


def generate_frames_cv2(filename: str | Path) -> Generator[tuple[int, np.ndarray], None, None]:
    video_capture = cv2.VideoCapture(filename, cv2.CAP_ANY)
    success = True
    frame_no = 1
    while success:
        success, array = video_capture.read()
        if success:
            yield frame_no, array
        frame_no += 1


class Florence2:
    def __init__(self, model_id="microsoft/Florence-2-large-ft", device: str = "cpu"):
        # "microsoft/Florence-2-large-ft"
        # "microsoft/Florence-2-base-ft"
        logger.info(f"Loading {model_id}")
        self.device = device
        self.torch_dtype = torch.float16 if self.device == "cuda" else torch.float32

        self.model = (
            AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
            .eval()
            .to(self.device, self.torch_dtype)
        )
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        logger.info(f"Loaded {model_id}")

    @torch.no_grad()
    def process(self, image: Image, task_prompt: str, text_input=None):
        prompt = task_prompt if text_input is None else task_prompt + text_input

        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device, self.torch_dtype)
        input_ids = inputs["input_ids"].to(self.device)
        pixel_values = inputs["pixel_values"].to(self.device)

        generated_ids = self.model.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            max_new_tokens=1024,
            early_stopping=False,
            do_sample=False,
            num_beams=3,
        )
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        return self.processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=(image.width, image.height),
        )

    def process_od(self, image: Image) -> Generator[tuple[str, tuple[int, int, int, int]], None, None]:
        results = self.process(
            image=image,
            task_prompt="<OD>",
        )["<OD>"]
        for label, bbox in zip(results["labels"], results["bboxes"], strict=False):
            bbox_pixel = (floor(bbox[0]), floor(bbox[1]), ceil(bbox[2]), ceil(bbox[3]))
            yield label, bbox_pixel

    def process_open_vocab(
        self, image: Image, text_input: str
    ) -> Generator[tuple[str, tuple[int, int, int, int]], None, None]:
        results = self.process(
            image=image,
            task_prompt="<OPEN_VOCABULARY_DETECTION>",
            text_input=text_input,
        )["<OPEN_VOCABULARY_DETECTION>"]
        for label, bbox in zip(results["bboxes_labels"], results["bboxes"], strict=False):
            bbox_pixel = (floor(bbox[0]), floor(bbox[1]), ceil(bbox[2]), ceil(bbox[3]))
            yield label, bbox_pixel


def main(filenames: Iterable[str | Path], device: str = "cpu"):
    florence = Florence2(device=device)

    for filename in filenames:
        filename = Path(filename)

        for frame_no, frame in generate_frames_cv2(filename):
            if (frame_no - 1) % 10 != 0:
                continue
            startime = time.time()
            logger.info(f"Processing frame {frame_no}")
            image = Image.fromarray(frame)

            for label, bbox in florence.process_od(image):
                logger.info(f"{frame_no} : {label} : {bbox}")

            endtime = time.time()
            logger.info(f"Processed frame {frame_no} in {endtime - startime}s")


app = typer.Typer()


@app.command()
def setup(filenames: list[str], device: str = "cpu") -> None:
    logging.basicConfig(level=logging.INFO)

    main(filenames, device)


if __name__ == "__main__":
    app()
