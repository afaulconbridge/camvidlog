import gc
import logging
from multiprocessing import Queue
from typing import Any

import cv2
import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from transformers import (
    AutoModelForImageSegmentation,
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
    CLIPModel,
    CLIPProcessor,
    GroundingDinoProcessor,
)

from camvidlog.procs.basics import Colourspace, DataRecorder, FrameConsumer, FrameConsumerProducer, FrameQueueInfoOutput
from camvidlog.procs.frame import split_frame
from camvidlog.queues import SharedMemoryQueueManager

logger = logging.getLogger(__name__)


class GroundingDino(FrameConsumer):
    processor: AutoProcessor
    text_queries: str
    processor_input_ids: Any
    queue_results: Queue

    def __init__(
        self,
        info_input: FrameQueueInfoOutput,
        queries: tuple[str, ...],
        data_recorder: DataRecorder,
        model_id: str,
        box_threshold=0.25,
        text_threshold=0.25,
        supplementary: dict[str, str] | None = None,
    ):
        super().__init__(
            info_input=info_input,
        )
        # will only accept a single string
        self.text_queries = ". ".join(queries) + "."
        self.model_id = model_id
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.supplementary = supplementary if supplementary else {}
        columns = (
            "hits.count",
            "hits.0.score",
            "hits.0.label",
            *self.supplementary.keys(),
        )
        self.queue_results = data_recorder.register(columns)

    def setup(self) -> None:
        self.processor: GroundingDinoProcessor = AutoProcessor.from_pretrained(self.model_id)
        # process the text input now, as it doesn't change frame-to-frame
        self.processor_input_ids = self.processor(text=self.text_queries, return_tensors="pt").to("cuda")
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id).to("cuda")

    def process_frame(self, frame_in) -> None:
        image_pillow = Image.fromarray(frame_in)
        inputs = self.processor(images=image_pillow, return_tensors="pt").to("cuda")
        outputs = self.model(**inputs, **self.processor_input_ids)
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            self.processor_input_ids.input_ids,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[image_pillow.size[::-1]],
        )[0]
        # score, label, bbox
        hits = tuple(zip(*results.values(), strict=False))
        for i, hit in enumerate(hits):
            score, label, bbox = hit
        metrics = {}
        metrics["hits.count"] = len(hits)
        if hits:
            score, label, bbox = hits[0]
            metrics["hits.0.score"] = float(score)
            metrics["hits.0.label"] = str(label)
            # yield float(score), str(label), [int(i) for i in bbox]
        metrics.update(self.supplementary)
        self.queue_results.put((self.frame_no, metrics))
        logger.info(f"Processed frame {self.frame_no}")

        for key, value in sorted(metrics.items()):
            logger.debug(f"{key} = {value}")

        # cleanup to preserve GPU memory
        del inputs
        del outputs
        del results
        gc.collect()
        torch.cuda.empty_cache()

        return True

    def close(self) -> None:
        super().close()

        self.queue_results.put(None)


class Clip(FrameConsumer):
    """
    CLIP is a transformer model. It maps image and the given text labels into the same latent space
    and then compares distance.

    Text is tokenized into word roots (kinda)
    Image is resized to a small size (typically 224x224x3) and then split into even smaller patches (14 or 16 or 32) to
    generate tokens.
    """

    processor: CLIPProcessor
    text_queries: str
    processor_input_ids: Any
    queue_results: Queue

    def __init__(
        self,
        info_input: FrameQueueInfoOutput,
        queries: tuple[str, ...],
        data_recorder: DataRecorder,
        model_id: str,
        supplementary: dict[str, str] | None = None,
    ):
        super().__init__(
            info_input=info_input,
        )
        # will only accept a single string
        self.text_queries = queries
        self.model_id = model_id
        self.supplementary = supplementary if supplementary else {}
        self.supplementary["model_id"] = model_id
        columns = ["score", "label"]
        columns.extend(self.supplementary.keys())
        self.queue_results = data_recorder.register(columns)

    def setup(self) -> None:
        self.processor = CLIPProcessor.from_pretrained(self.model_id)
        # process the text input now, as it doesn't change frame-to-frame
        self.processor_input_ids = self.processor(
            text=self.text_queries,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to("cuda")
        self.model = CLIPModel.from_pretrained(self.model_id).to("cuda")

    def process_frame(self, frame_in) -> None:
        image_pillow = Image.fromarray(frame_in)
        inputs = self.processor(
            text=self.text_queries,
            images=image_pillow,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to("cuda")
        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
        results = dict(zip(self.text_queries, tuple(float(i) for i in probs[0]), strict=False))

        for label, score in results.items():
            metrics = {"label": label, "score": score}
            metrics.update(self.supplementary)
            self.queue_results.put((self.frame_no, metrics))
        logger.info(f"Processed frame {self.frame_no}")

        return True

    def close(self) -> None:
        super().close()

        self.queue_results.put(None)


class OpenClip(FrameConsumer):
    """
    OpenClip is library for Clip transformer model. It maps image and the given text labels into the same latent space
    and then compares distance.

    Text is tokenized into word roots (kinda)
    Image is resized to a small size (typically 224x224x3) and then split into even smaller patches (14 or 16 or 32) to
    generate tokens.
    """

    processor: CLIPProcessor
    text_queries: str
    processor_input_ids: Any
    queue_results: Queue
    background_subtractor: cv2.BackgroundSubtractor | None = None

    def __init__(
        self,
        info_input: FrameQueueInfoOutput,
        queries: tuple[str, ...],
        data_recorder: DataRecorder,
        supplementary: dict[str, str] | None = None,
    ):
        super().__init__(
            info_input=info_input,
        )
        self.text_queries = queries
        self.model_id = "imageomics/bioclip"
        self.supplementary = supplementary if supplementary else {}
        self.supplementary["model_id"] = self.model_id
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=900, detectShadows=False
        )  # TODO customize

        columns = ["score", "label"]
        columns.extend(self.supplementary.keys())
        self.queue_results = data_recorder.register(columns)

    def setup(self) -> None:
        self.model, _, self.processor = open_clip.create_model_and_transforms(
            "hf-hub:imageomics/bioclip",
            device=torch.device("cuda"),
        )
        self.model.eval()
        tokenizer = open_clip.get_tokenizer(
            "hf-hub:imageomics/bioclip",
        )
        with torch.no_grad():
            self.text_features = F.normalize(self.model.encode_text(tokenizer(self.text_queries).to("cuda")), dim=-1)

    def process_frame(self, frame_in) -> None:
        self.background_subtractor.apply(frame_in)
        frame_pil = Image.fromarray(frame_in.squeeze())
        bg_pil = Image.fromarray(self.background_subtractor.getBackgroundImage().squeeze())
        # bg_pil = Image.new("L", frame_pil.size)
        # frame_pil = frame_pil.crop(frame_pil.getbbox())
        with torch.no_grad():
            image_features = F.normalize(
                self.model.encode_image(self.processor(frame_pil).to("cuda").unsqueeze(0)), dim=-1
            )
            image_features = image_features.clone().detach()

            bg_features = F.normalize(self.model.encode_image(self.processor(bg_pil).to("cuda").unsqueeze(0)), dim=-1)
            bg_features = bg_features.clone().detach()

            image_text_probs = image_features @ self.text_features.T
            bg_text_probs = bg_features @ self.text_features.T
            text_probs = (image_text_probs / bg_text_probs).softmax(dim=-1)
            results = dict(zip(self.text_queries, [float(i) for i in text_probs[0]], strict=True))

        for label, score in results.items():
            metrics = {"label": label, "score": score}
            metrics.update(self.supplementary)
            self.queue_results.put((self.frame_no, metrics))
        logger.info(f"Processed frame {self.frame_no}")

        return True

    def close(self) -> None:
        super().close()

        self.queue_results.put(None)


class ClipSplitter(FrameConsumer):
    """
    CLIP is a transformer model. It maps image and the given text labels into the same latent space
    and then compares distance.

    Text is tokenized into word roots (kinda)
    Image is chopped into small size chunks (typically 224x224x3) then split into patches (14 or 16 or 32) to
    generate tokens.
    """

    processor: CLIPProcessor
    text_queries: str
    processor_input_ids: Any
    queue_results: Queue

    def __init__(
        self,
        info_input: FrameQueueInfoOutput,
        queries: tuple[str, ...],
        data_recorder: DataRecorder,
        model_id: str,
        supplementary: dict[str, str] | None = None,
    ):
        super().__init__(
            info_input=info_input,
        )
        # will only accept a single string
        self.text_queries = queries
        self.model_id = model_id
        self.supplementary = supplementary if supplementary else {}
        self.supplementary["model_id"] = model_id
        columns = ["score", "label", "i", "j", "subsize"]
        columns.extend(self.supplementary.keys())
        self.queue_results = data_recorder.register(columns)

    def setup(self) -> None:
        self.processor = CLIPProcessor.from_pretrained(self.model_id)
        # process the text input now, as it doesn't change frame-to-frame
        self.processor_input_ids = self.processor(
            text=self.text_queries,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to("cuda")
        self.model = CLIPModel.from_pretrained(self.model_id).to("cuda")

    def process_frame(self, frame_in) -> None:
        for subsize in (0, 336 * 4, 336 * 2):  # , 336
            if subsize == 0:
                frame_splits = [((0, 0), frame_in)]
            else:
                frame_splits = split_frame(frame_in, subsize)

            for (i, j), frame_split in frame_splits:
                inputs = self.processor(
                    text=self.text_queries,
                    images=frame_split,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                ).to("cuda")
                outputs = self.model(**inputs)
                logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
                probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
                results = dict(zip(self.text_queries, tuple(float(i) for i in probs[0]), strict=False))

                for label, score in results.items():
                    metrics = {"label": label, "score": score, "i": i, "j": j, "subsize": subsize}
                    metrics.update(self.supplementary)
                    self.queue_results.put((self.frame_no, metrics))
        logger.info(f"Processed frame {self.frame_no}")

        return True

    def close(self) -> None:
        super().close()

        self.queue_results.put(None)


class BiRefNet(FrameConsumerProducer):
    """
    BiRefNet is for generating a mask of foreground objects for background removal

    Input is a 1024x1024 image
    Output is a boolean mask
    """

    processor: Any
    model_id: str

    def __init__(
        self,
        info_input: FrameQueueInfoOutput,
        queue_manager: SharedMemoryQueueManager,
        model_id: str = "ZhengPeng7/BiRefNet",
    ):
        super().__init__(info_input=info_input, queue_manager=queue_manager)
        if info_input.x != 1024 or info_input.y != 1024:  # noqa: PLR2004
            msg = "Input image must be 1024x1024"
            raise RuntimeError(msg)
        # will only accept a single string
        self.model_id = model_id
        # always outputs boolean
        self.info_output.colourspace = Colourspace.greyscale

    def setup(self) -> None:
        self.processor = AutoModelForImageSegmentation.from_pretrained("ZhengPeng7/BiRefNet", trust_remote_code=True)
        torch.set_float32_matmul_precision(["high", "highest"][0])
        self.processor.to("cuda")
        self.processor.eval()

    def process_frame(self, frame_in: np.ndarray, frame_out: np.ndarray) -> bool:
        input_images = torch.from_numpy(frame_in.transpose((2, 0, 1))).contiguous().div(255)
        input_images = input_images.unsqueeze(0).to("cuda")

        # Prediction
        with torch.no_grad():
            preds = self.processor(input_images)[-1].sigmoid().cpu()
        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred).convert("L")

        image = Image.frombytes("RGB", (self.info_input.x, self.info_input.y), frame_in)

        image_out = Image.composite(image, Image.new("L", image.size), pred_pil)
        np_out = np.expand_dims(image_out, axis=2)
        np.copyto(frame_out, np_out)
        return True
