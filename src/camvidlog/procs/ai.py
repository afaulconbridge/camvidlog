from multiprocessing import Queue
from typing import Any

import numpy as np
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, GroundingDinoProcessor

from camvidlog.procs.basics import FrameConsumer, FrameQueueInfoOutput


class GroundingDino(FrameConsumer):
    processor: AutoProcessor
    text_queries: str
    processor_input_ids: Any
    queue_results: Queue

    def __init__(
        self,
        info_input: FrameQueueInfoOutput,
        queries: tuple[str, ...],
        queue_results: Queue,
        model_id: str,
        box_threshold=0.25,
        text_threshold=0.25,
    ):
        super().__init__(
            info_input=info_input,
        )
        self.queue_results = queue_results
        # will only accept a single string
        self.text_queries = ". ".join(queries) + "."
        self.model_id = model_id
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

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
        if hits:
            score, label, bbox = hits[0]
            print((float(score), str(label)))
            self.queue_results.put((self.frame_no, "hits.0.score", float(score)))
            self.queue_results.put((self.frame_no, "hits.0.label", str(label)))
            # yield float(score), str(label), [int(i) for i in bbox]
        return True

    def cleanup(self) -> None:
        super().cleanup()

        self.queue_results.put(None)
