import gc
import logging
from multiprocessing import Queue
from typing import Any

import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, GroundingDinoProcessor

from camvidlog.procs.basics import DataRecorder, FrameConsumer, FrameQueueInfoOutput

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
        #        for i, hit in enumerate(hits):
        #            score, label, bbox = hit
        #            print((i, float(score), str(label)))
        metrics = {}
        if hits:
            score, label, bbox = hits[0]
            metrics["hits.0.score"] = float(score)
            metrics["hits.0.label"] = str(label)
            # yield float(score), str(label), [int(i) for i in bbox]
        metrics.update(self.supplementary)
        self.queue_results.put((self.frame_no, metrics))
        logger.info(f"Processed frame {self.frame_no}")
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
