import logging
import statistics
from collections.abc import Generator
from pathlib import Path
from subprocess import Popen
from typing import Self

import cv2
import ffmpeg
import numpy as np
import open_clip
import torch
from cv2.typing import MatLike
from PIL import Image
from pydantic import NonNegativeInt, model_validator
from pydantic.dataclasses import dataclass
from transformers import (
    CLIPProcessor,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FrameFragment:
    x_min: NonNegativeInt
    y_min: NonNegativeInt
    x_max: NonNegativeInt
    y_max: NonNegativeInt

    @model_validator(mode="after")
    def check_size(self) -> Self:
        if self.x_max <= self.x_min:
            msg = "No width"
            raise ValueError(msg)
        if self.y_max <= self.y_min:
            msg = "No height"
            raise ValueError(msg)


@dataclass(frozen=True)
class QueryTerm:
    label: str
    prompt: str
    weight: float = 1.0


@dataclass(frozen=True)
class FrameFragmentResult:
    label: QueryTerm
    score: float
    fragment: FrameFragment


def generate_fragments(x: int, y: int, subsize: int) -> Generator[FrameFragment, None, None]:
    if x == subsize and y == subsize:
        yield FrameFragment(0, 0, x, y)
    else:
        # for each quadrant
        # work out size to fill
        gap_x = (x // 2) + (subsize // 2)
        gap_y = (y // 2) + (subsize // 2)
        count_x = (gap_x // subsize) + 1
        count_y = (gap_y // subsize) + 1
        patch_gap_x = gap_x // count_x
        patch_gap_y = gap_y // count_y

        for i in range(count_x + count_x - 1):
            offset_x = patch_gap_x * i
            for j in range(count_y + count_y - 1):
                offset_y = patch_gap_y * j
                yield FrameFragment(offset_x, offset_y, offset_x + subsize, offset_y + subsize)

        # continue with larger fragments
        if subsize < min(x, y) / 3:
            yield from generate_fragments(x, y, subsize * 2)


def split_frame(frame_in: MatLike, subsize: int) -> Generator[tuple[FrameFragment, np.ndarray], None, None]:
    y, x, _ = frame_in.shape
    for fragment in generate_fragments(x, y, subsize):
        subimage = frame_in[fragment.y_min : fragment.y_max, fragment.x_min : fragment.x_max]
        yield fragment, subimage


class OpenClip:
    preprocessor: CLIPProcessor
    queries: tuple[QueryTerm, ...]

    def __init__(
        self,
        queries: tuple[QueryTerm, ...],
    ):
        self.queries = queries
        self.model_id = "imageomics/bioclip"

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
            text_tokens = tokenizer(q.prompt for q in self.queries).to("cuda")
            self.text_features = self.model.encode_text(text_tokens).float()
            self.text_features /= self.text_features.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def process_frame(self, frame_in: MatLike, bg: MatLike) -> Generator[FrameFragmentResult, None, None]:
        frames = []
        bgs = []
        fragments = []

        for (
            fragment,
            sub_frame,
        ), (
            fragment_bg,
            sub_bg,
        ) in zip(
            split_frame(frame_in=frame_in, subsize=384),
            split_frame(frame_in=bg, subsize=384),
            strict=True,
        ):
            if fragment != fragment_bg:
                raise RuntimeError("Different fragments detected")

            frame_pil = Image.fromarray(sub_frame.squeeze())
            bg_pil = Image.fromarray(sub_bg.squeeze())
            frames.append(self.preprocessor(frame_pil))
            bgs.append(self.preprocessor(bg_pil))
            fragments.append(fragment)

        frame_features = self.model.encode_image(torch.tensor(np.stack(frames)).to("cuda"))
        frame_features /= frame_features.norm(dim=-1, keepdim=True)
        frame_text_probs = frame_features @ self.text_features.T
        bg_features = self.model.encode_image(torch.tensor(np.stack(bgs)).to("cuda"))
        bg_features /= bg_features.norm(dim=-1, keepdim=True)
        bg_text_probs = bg_features @ self.text_features.T

        text_result = (frame_text_probs / bg_text_probs).to("cpu")
        for sub_result, fragment in zip(
            text_result,
            fragments,
            strict=True,
        ):
            for label, score in zip(self.queries, sub_result, strict=False):
                yield FrameFragmentResult(label, float(score), fragment)


def generate_frames_ffmpeg(filename) -> Generator[np.ndarray, None, None]:
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
        logger.info("Terminating")
        reader.terminate()
        logger.info("Terminated")


def generate_frames_cv2(filename) -> Generator[np.ndarray, None, None]:
    video_capture = cv2.VideoCapture(filename, cv2.CAP_ANY)
    success = True
    while success:
        success, array = video_capture.read()
        if success:
            yield array


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


def make_background_files(filename: str, filename_bg_avg: str, filename_bg_mog: str):
    logger.info(f"Generating background files {filename_bg_avg} & {filename_bg_mog}")

    # TODO customize history
    background_subtractor = cv2.createBackgroundSubtractorMOG2(history=300, detectShadows=False)

    average = None
    for frame_no, frame in enumerate(generate_frames_cv2(filename), start=1):
        frame_float = frame.astype(np.float32)
        if average is None:
            average = np.full(frame.shape, 128.0, np.float32)
        else:
            average = add_average(average, frame_float, frame_no)

        background_subtractor.apply(frame)

    bg_avg = average.astype(np.uint8)
    Image.fromarray(bg_avg, mode="RGB").save(filename_bg_avg)

    bg_mog = background_subtractor.getBackgroundImage()
    Image.fromarray(bg_mog, mode="RGB").save(filename_bg_mog)
    logger.info(f"Generated background files {filename_bg_avg} & {filename_bg_mog}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    filename = Path("/workspaces/camvidlog/data/20231117214546_VD_00327.MP4")
    filename_bg_avg = filename.with_stem(f"{filename.stem}.bg.avg").with_suffix(".png")
    filename_bg_mog = filename.with_stem(f"{filename.stem}.bg.mog").with_suffix(".png")
    if not filename_bg_avg.exists() or not filename_bg_mog.exists():
        make_background_files(filename, filename_bg_avg, filename_bg_mog)

    queries = [
        QueryTerm(
            "deer",
            "Eukaryota Animalia Chordata Mammalia Artiodactyla Cervidae Cervinae Muntiacini Muntiacus (muntjac deer)",
        ),
        QueryTerm(
            "cat",
            "Eukaryota Animalia Chordata Mammalia Carnivora Feliformia Felidae Felinae Felis Felis catus (domestic cat)",
        ),
        QueryTerm(
            "hedgehog",
            "Eukaryota Animalia Chordata Mammalia Eulipotyphla Erinaceidae Erinaceus Erinaceus europaeus (European hedgehog)",
        ),
        QueryTerm(
            "fox",
            "Eukaryota Animalia Chordata Mammalia Carnivora Canidae Vulpes Vulpes vulpes (red fox)",
        ),
        QueryTerm(
            "otter",
            "otter",
        ),
        QueryTerm(
            "mink",
            "mink",
        ),
        QueryTerm(
            "badger",
            "badger",
        ),
        QueryTerm(
            "rat",
            "rat",
        ),
        QueryTerm(
            "mouse",
            "mouse",
        ),
        QueryTerm(
            "mole",
            "mole",
        ),
    ]
    # ferret

    clip = OpenClip(queries)
    clip.setup()

    background_subtractor = cv2.createBackgroundSubtractorMOG2(history=900, detectShadows=False)

    label_scores = {}
    label_statuses = {}

    for frame_no, frame in enumerate(generate_frames_cv2(filename), start=1):
        logger.info(f"Processing frame {frame_no}")
        background_subtractor.apply(frame)
        bg = background_subtractor.getBackgroundImage()

        fragment_results = clip.process_frame(frame, bg)
        # keep the highest score for each label
        label_fragment_max = {}
        for fragment in fragment_results:
            if fragment.label not in label_fragment_max or fragment.score > label_fragment_max[fragment.label].score:
                label_fragment_max[fragment.label] = fragment

        # build up a list of scores for each label over time
        for label, fragment in label_fragment_max.items():
            if label not in label_scores:
                label_scores[label] = []
            label_scores[label].append(fragment)

        # look at the rolling median over the last N frames
        label_medians = {}
        n = 30
        for label, fragments in label_scores.items():
            if len(fragments) < n:
                continue
            score_median = statistics.median(f.score for f in fragments[-n:])
            score_median_corrected = score_median * label.weight
            label_medians[label] = score_median_corrected

        # start an event when median goes above X
        # stop an even when it goes below Y
        start_value = 1.4
        stop_value = 1.2

        for label, label_median in label_medians.items():
            if label_statuses.get(label, 0):
                # already activated, check to stop
                if label_median < stop_value:
                    start_at = label_statuses[label]
                    end_at = frame_no
                    label_statuses[label] = 0
                    # TODO store and output at end
                    print(f"{label.label} - {start_at} - {end_at}")
            elif label_median > start_value:
                label_statuses[label] = frame_no
