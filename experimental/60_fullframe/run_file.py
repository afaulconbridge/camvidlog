import csv
import functools
import gzip
import io
import logging
from collections.abc import Generator, Iterable
from pathlib import Path
from typing import Self

import cv2
import numpy as np
import open_clip
import torch
import typer
from cv2.typing import MatLike
from PIL import Image
from pydantic import BaseModel, ConfigDict, NonNegativeInt, model_validator
from torchvision import transforms
from transformers import (
    CLIPProcessor,
)

logger = logging.getLogger(__name__)


class QueryTerm(BaseModel):
    model_config = ConfigDict(frozen=True)

    label: str
    prompt: str
    weight: float = 1.0


class FrameResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    ai_model: str
    label: QueryTerm
    score: float
    score_bg: float
    score_frame: float


class OpenClip:
    preprocessor: CLIPProcessor
    queries: tuple[QueryTerm, ...]
    model_id: str

    preprocess_img = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((224, 224), antialias=True),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )

    def __init__(
        self,
        model_id: str,
        queries: tuple[QueryTerm, ...],
    ):
        self.queries = queries
        self.model_id = model_id

    def setup(self) -> None:
        self.model, _, self.preprocessor = open_clip.create_model_and_transforms(
            self.model_id,
        )
        self.model.eval()
        # self.model.save_pretrained("pretrained")
        tokenizer = open_clip.get_tokenizer(
            self.model_id,
        )
        with torch.no_grad():
            text_tokens = tokenizer(q.prompt for q in self.queries)
            # this is the hard compute part!
            self.text_features = self.model.encode_text(text_tokens).float()
            # rescale to length of one
            self.text_features /= self.text_features.norm(dim=-1, keepdim=True)

    def get_frame_features(self, frame: MatLike) -> torch.Tensor:
        frame_tensor = self.preprocessor(Image.fromarray(frame)).unsqueeze(0)
        # this is the hard compute part!
        frame_features = self.model.encode_image(frame_tensor)

        return frame_features

    def get_frame_text_score(self, frame_features: torch.Tensor) -> torch.Tensor:
        # rescale to length of one
        frame_features /= frame_features.norm(dim=-1, keepdim=True)

        # calculate dot product against the text features
        frame_text_probs = frame_features @ self.text_features.T

        # rescale from -inf---inf to 0---1
        frame_text_probs = frame_text_probs.softmax(dim=-1)

        return frame_text_probs[0]  # only had one frame to handle

    def get_frame_text_score_compared_to_background(
        self,
        frame_features: torch.Tensor,
        background_text_dot_product: torch.Tensor,
    ) -> torch.Tensor:
        # rescale to length of one
        frame_features /= frame_features.norm(dim=-1, keepdim=True)

        # calculate dot product against the text features
        frame_text_dot_product = frame_features @ self.text_features.T

        # calculate what propoprtion of the total dot product was to the frame
        proportional_dot_product = frame_text_dot_product / (frame_text_dot_product + background_text_dot_product)

        return proportional_dot_product[0]  # only had one frame to handle

    def get_frame_text_score_subtracting_background(
        self,
        frame_features: torch.Tensor,
        background_features: torch.Tensor,
    ) -> torch.Tensor:
        # rescale to length of one
        frame_features /= frame_features.norm(dim=-1, keepdim=True)

        # subtract the background
        frame_fg_features = frame_features - background_features

        # rescale to length of one
        frame_fg_features /= frame_fg_features.norm(dim=-1, keepdim=True)

        # calculate dot product against the text features
        frame_text_probs = frame_fg_features @ self.text_features.T

        # rescale from -inf---inf to 0---1
        frame_text_probs = frame_text_probs.softmax(dim=-1)

        return frame_text_probs[0]  # only had one frame to handle


def generate_frames_cv2(filename: str | Path) -> Generator[tuple[int, np.ndarray], None, None]:
    video_capture = cv2.VideoCapture(filename, cv2.CAP_ANY)
    success = True
    frame_no = 1
    while success:
        success, array = video_capture.read()
        if success:
            yield frame_no, array
        frame_no += 1


def make_background_files(filename: str, filename_bg_mog: str):
    logger.info(f"Generating background files {filename_bg_mog}")

    # TODO customize history
    background_subtractor = cv2.createBackgroundSubtractorMOG2(history=300, detectShadows=False)

    for _, frame in generate_frames_cv2(filename):
        background_subtractor.apply(frame)

    bg_mog = background_subtractor.getBackgroundImage()
    Image.fromarray(bg_mog, mode="RGB").save(filename_bg_mog)
    logger.info(f"Generated background files {filename_bg_mog}")


def main(filenames: Iterable[str | Path]):
    queries = [
        QueryTerm(
            label="deer",
            # prompt="Eukaryota Animalia Chordata Mammalia Artiodactyla Cervidae Cervinae Muntiacini Muntiacus (muntjac deer)",
            prompt="deer",
        ),
        QueryTerm(
            label="cat",
            # prompt="Eukaryota Animalia Chordata Mammalia Carnivora Feliformia Felidae Felinae Felis Felis catus (domestic cat)",
            prompt="cat",
        ),
        QueryTerm(
            label="hedgehog",
            # prompt="Eukaryota Animalia Chordata Mammalia Eulipotyphla Erinaceidae Erinaceus Erinaceus europaeus (European hedgehog)",
            prompt="hedgehog",
        ),
        QueryTerm(
            label="fox",
            # prompt="Eukaryota Animalia Chordata Mammalia Carnivora Canidae Vulpes Vulpes vulpes (red fox)",
            prompt="fox",
        ),
        QueryTerm(
            label="otter",
            prompt="otter",
        ),
        QueryTerm(
            label="mink",
            prompt="mink",
        ),
        QueryTerm(
            label="badger",
            prompt="badger",
        ),
        QueryTerm(
            label="rat",
            prompt="rat",
        ),
        QueryTerm(
            label="mouse",
            prompt="mouse",
        ),
        QueryTerm(
            label="mole",
            prompt="mole",
        ),
    ]
    # ferret

    for filename in filenames:
        filename = Path(filename)
        filename_bg_mog = filename.with_stem(f"{filename.stem}.bg.mog").with_suffix(".png")
        if not filename_bg_mog.exists():
            make_background_files(filename, filename_bg_mog)

        bg_img = cv2.imread(filename_bg_mog)

        with io.TextIOWrapper(gzip.GzipFile(filename.with_suffix(".csv.gz"), "wb"), encoding="utf-8") as outfile:
            csvwriter = csv.DictWriter(
                outfile,
                fieldnames=["model_id", "frame_no", "label", "score", "score_bg", "score_bg_sub"],
            )
            csvwriter.writeheader()

            model_ids = [
                "hf-hub:imageomics/bioclip",
                # these exhibit little frame-to-frame variance suggesting they don't recognise anything
                # "ViT-S-32",
                # "ViT-M-32",
                # "ViT-B-32",
                # "ViT-H-14-378-quickgelu",
            ]
            for model_id in model_ids:
                clip = OpenClip(model_id, queries)
                clip.setup()

                bg_features = clip.get_frame_features(bg_img)
                # rescale to length of one
                bg_features /= bg_features.norm(dim=-1, keepdim=True)
                # calculate dot product against the text features
                background_text_dot_product = bg_features @ clip.text_features.T

                with torch.no_grad():
                    for frame_no, frame in generate_frames_cv2(filename):
                        #                        if frame_no % 10 != 1:
                        #                            continue
                        logger.info(f"Processing frame {frame_no}")

                        frame_features = clip.get_frame_features(frame)

                        frame_text_scores = clip.get_frame_text_score(frame_features)
                        frame_text_bg_scores = clip.get_frame_text_score_compared_to_background(
                            frame_features, background_text_dot_product
                        )
                        frame_text_bgsub_scores = clip.get_frame_text_score_subtracting_background(
                            frame_features, bg_features
                        )

                        for query, frame_text_score, frame_text_bg_score, frame_text_bg_sub_score in zip(
                            queries,
                            frame_text_scores,
                            frame_text_bg_scores,
                            frame_text_bgsub_scores,
                            strict=True,
                        ):
                            csvwriter.writerow(
                                {
                                    "model_id": model_id,
                                    "frame_no": frame_no,
                                    "label": query.label,
                                    "score": float(frame_text_score),
                                    "score_bg": float(frame_text_bg_score),
                                    "score_bg_sub": float(frame_text_bg_sub_score),
                                }
                            )


app = typer.Typer()


@app.command()
def setup(filenames: list[str]) -> None:
    logging.basicConfig(level=logging.INFO)

    main(filenames)


if __name__ == "__main__":
    app()
