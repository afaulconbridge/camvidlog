import logging
from collections.abc import Generator, Iterator
from dataclasses import dataclass
from subprocess import Popen

import cv2
import ffmpeg
import numpy as np
import open_clip
import torch
from cv2.typing import MatLike
from PIL import Image
from transformers import (
    CLIPProcessor,
)

logger = logging.getLogger(__name__)


@dataclass
class FrameFragmentResult:
    label: int
    score: float
    x_min: int
    x_max: int
    y_min: int
    y_max: int


def split_frame(
    frame_in: MatLike, subsize: int
) -> Generator[tuple[tuple[int, int], tuple[int, int], np.ndarray], None, None]:
    y, x, _ = frame_in.shape
    # always have a central subimage
    if x == subsize and y == subsize:
        yield (0, 0), (x, y), frame_in
        return
    # for each quadrant
    # work out size to fill
    gap_x = (x // 2) + (subsize // 2)
    gap_y = (y // 2) + (subsize // 2)
    count_x = (gap_x // subsize) + 1
    count_y = (gap_y // subsize) + 1
    patch_gap_x = gap_x // count_x
    patch_gap_y = gap_y // count_y

    for i in range(count_x + count_x - 1):
        for j in range(count_y + count_y - 1):
            offset_x = patch_gap_x * i
            offset_y = patch_gap_y * j
            subimage = frame_in[offset_y : offset_y + subsize, offset_x : offset_x + subsize]
            yield (offset_x, offset_y), (offset_x + subsize, offset_y + subsize), subimage


class OpenClip:
    preprocessor: CLIPProcessor
    text_queries: str
    background_subtractor: cv2.BackgroundSubtractor | None = None

    def __init__(
        self,
        queries: tuple[str, ...],
    ):
        self.text_queries = queries
        self.model_id = "imageomics/bioclip"
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=50, detectShadows=False
        )  # TODO customize

    def setup(self) -> None:
        self.model, _, self.preprocessor = open_clip.create_model_and_transforms(
            "hf-hub:imageomics/bioclip",
            device=torch.device("cuda"),
        )
        self.model.eval()
        tokenizer = open_clip.get_tokenizer(
            "hf-hub:imageomics/bioclip",
        )
        with torch.no_grad():
            text_tokens = tokenizer(self.text_queries).to("cuda")
            self.text_features = self.model.encode_text(text_tokens).float()
            self.text_features /= self.text_features.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def process_frame(self, frame_in: MatLike, bg: MatLike) -> Generator[FrameFragmentResult, None, None]:
        self.background_subtractor.apply(frame_in)
        bg = self.background_subtractor.getBackgroundImage()
        frames = []
        bgs = []
        positions = []

        for (
            (split_x_min, split_y_min),
            (split_x_max, split_y_max),
            sub_frame,
        ), (
            (bgsplit_x_min, bgsplit_y_min),
            (bgsplit_x_max, bgsplit_y_max),
            sub_bg,
        ) in zip(
            split_frame(frame_in=frame_in, subsize=384),
            split_frame(frame_in=bg, subsize=384),
            strict=True,
        ):
            assert split_x_min == bgsplit_x_min
            assert split_y_min == bgsplit_y_min
            assert split_x_max == bgsplit_x_max
            assert split_y_max == bgsplit_y_max

            frame_pil = Image.fromarray(sub_frame.squeeze())
            bg_pil = Image.fromarray(sub_bg.squeeze())
            frames.append(self.preprocessor(frame_pil))
            bgs.append(self.preprocessor(bg_pil))
            positions.append(((split_x_min, split_y_min), (split_x_max, split_y_max)))

        frame_features = self.model.encode_image(torch.tensor(np.stack(frames)).to("cuda"))
        frame_features /= frame_features.norm(dim=-1, keepdim=True)
        frame_text_probs = frame_features @ self.text_features.T
        bg_features = self.model.encode_image(torch.tensor(np.stack(bgs)).to("cuda"))
        bg_features /= bg_features.norm(dim=-1, keepdim=True)
        bg_text_probs = bg_features @ self.text_features.T

        text_result = (frame_text_probs / bg_text_probs).to("cpu")
        for sub_result, (
            (split_x_min, split_y_min),
            (split_x_max, split_y_max),
        ) in zip(
            text_result,
            positions,
            strict=True,
        ):
            for label, score in zip(self.text_queries, sub_result, strict=False):
                yield FrameFragmentResult(label, float(score), split_x_min, split_x_max, split_y_min, split_y_max)

        logger.info("Processed frame")

    def process_frames(self, frames_in: Iterator[MatLike]) -> Generator[FrameFragmentResult, None, None]:
        for frame in frames_in:
            yield self.process_frame(frame)


def generate_frames(filename) -> Generator[np.ndarray, None, None]:
    ffmpeg_kwargs = {}

    probe = ffmpeg.probe(filename, **ffmpeg_kwargs)
    video_stream = next(
        (stream for stream in probe["streams"] if stream["codec_type"] == "video"),
        None,
    )
    width = int(video_stream["width"])
    height = int(video_stream["height"])

    reader: Popen = (
        ffmpeg.input(
            filename,
            hwaccel="cuda",
            **ffmpeg_kwargs,
        )
        .output(
            "pipe:",
            format="rawvideo",
            pix_fmt="rgb24",
        )
        .run_async(
            pipe_stdout=True,
            quiet=True,
        )
    )

    try:
        while True:
            in_bytes = reader.stdout.read(width * height * 3)
            if len(in_bytes) != width * height * 3:
                break
            array = np.ndarray(
                (height, width, 3),
                dtype=np.uint8,
                buffer=in_bytes,
            )
            yield array
    finally:
        reader.terminate()


def add_average(average: np.ndarray, update: np.ndarray, count: int):
    if average.dtype != update.dtype:
        msg = "dtype of average and update must match"
        raise RuntimeError(msg)
    if update.dtype.kind != "f":
        msg = "dtpe of update must be float"
        raise RuntimeError(msg)
    update_delta = (update - average) / float(count)
    update_average = average + update_delta
    return update_average


def generate_average(filename, max_count=0) -> np.ndarray:
    average = None
    for frame_no, frame in enumerate(generate_frames(filename), start=1):
        frame_float = frame.astype(np.float32)
        if average is None:
            average = np.full(frame.shape, 128.0, np.float32)
        else:
            average = add_average(average, frame_float, frame_no)
        if max_count and frame_no >= max_count:
            break
    output = average.astype(np.uint8)
    return output


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    filename = "/workspaces/camvidlog/data/20231117214546_VD_00327.MP4"
    average_frame = generate_average(filename)
    Image.fromarray(average_frame, mode="RGB").save(f"{filename}.bg_avg.jpg")
    logging.info("Saved average background")
    1 / 0

    queries = [
        "muntjac deer",
        "domestic cat",
        "European hedgehog",
        "red fox",
        "otter",
        "mink",
        "badger",
        "domestic ferret",
        "rat",
        "mouse",
        "mole",
    ]

    clip = OpenClip(queries)
    clip.setup()

    for frame_no, array in enumerate(generate_frames(filename), start=1):
        fragment_results = clip.process_frame(array)
        # keep the highest score for each label
        label_fragment_max = {}
        for fragment in fragment_results:
            if fragment.label not in label_fragment_max or fragment.score > label_fragment_max[fragment.label].score:
                label_fragment_max[fragment.label] = fragment

        for label, fragment in sorted(label_fragment_max.items()):
            if fragment.score > 1.25:
                print(f"{frame_no} - {label} - {fragment.score}")
