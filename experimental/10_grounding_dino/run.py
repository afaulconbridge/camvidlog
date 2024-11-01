import logging
import time
from contextlib import contextmanager

import ffmpeg
import torch
from PIL import Image, ImageDraw
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@contextmanager
def timectx(description: str):
    start = time.time()
    yield
    end = time.time()
    logger.info(f"{description} = {(end-start)*1000:.2f}ms")


model_id = "IDEA-Research/grounding-dino-base"
device = "cuda"

# ffmpeg_src = "rtsp://192.168.1.104:8554/cam1_sub"
ffmpeg_src = "file:///workspaces/camvidlog/data/20231117215252_VD_00328.MP4"
ffmpeg_dst = "file:///workspaces/camvidlog/out.MP4"
text_queries = ["screenshot from trail camera video footage", "person", "cat", "deer", "bird"]

logger.info("setup started")

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

probe = ffmpeg.probe(ffmpeg_src)
video_stream = next(
    (stream for stream in probe["streams"] if stream["codec_type"] == "video"),
    None,
)
width = int(video_stream["width"])
height = int(video_stream["height"])

# -hwaccel cuda -hwaccel_output_format cuda
# -hwaccel qsv -c:v h264_qsv


reader = (
    ffmpeg.input(ffmpeg_src)
    .output("pipe:", format="rawvideo", pix_fmt="rgb24")
    .run_async(pipe_stdout=True, quiet=True)
)
writer = (
    ffmpeg.input(
        "pipe:",
        format="rawvideo",
        pix_fmt="rgb24",
        s=f"{width}x{height}",
    )
    .output(
        ffmpeg_dst,
        pix_fmt="yuv420p",
    )
    .overwrite_output()
    .run_async(
        pipe_stdin=True,
        quiet=True,
    )
)

logger.info("setup complete")
i = -1
while True:
    i += 1

    with timectx(f"frame {i} read"):
        in_bytes = reader.stdout.read(width * height * 3)
        if len(in_bytes) != width * height * 3:
            break
        image = Image.frombytes("RGB", (width, height), in_bytes)

    if i % 30 == 0:
        with timectx(f"frame {i} process"):
            inputs = processor(images=image, text=". ".join(text_queries), return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                results = processor.post_process_grounded_object_detection(
                    outputs, inputs.input_ids, box_threshold=0.5, text_threshold=0.5, target_sizes=[image.size[::-1]]
                )[0]

    with timectx(f"frame {i} write"):
        draw = ImageDraw.Draw(image)
        for j, (score, label, bbox) in enumerate(zip(*results.values(), strict=False)):
            if label.startswith("screens"):
                # screens or screenshot ?
                # skip the context label
                continue
            logging.info(f"{i} {j} {score} {label}")
            xmin, ymin, xmax, ymax = bbox
            draw.rectangle((xmin, ymin, xmax, ymax), outline="red", width=1)
            draw.text((xmin, ymin), f"{label} {score:0.2f}", fill="white")
        writer.stdin.write(image.tobytes())

writer.stdin.close()
writer.wait()

for line in reader.stderr.readlines():
    print(line)
