import ffmpeg
import numpy as np
from PIL import Image

ffmpeg_src = "data/20231117214546_VD_00327.MP4"


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


if __name__ == "__main__":
    probe = ffmpeg.probe(ffmpeg_src)
    video_stream = next(
        (stream for stream in probe["streams"] if stream["codec_type"] == "video"),
        None,
    )
    width = int(video_stream["width"])
    height = int(video_stream["height"])

    reader = (
        ffmpeg.input(ffmpeg_src)
        .output("pipe:", format="rawvideo", pix_fmt="rgb24")
        .run_async(pipe_stdout=True, quiet=True)
    )

    frame_no = -1
    shape = (height, width, 3)
    average = np.full(shape, 128.0, np.float32)
    average_count = 0
    average_max = 30 * 30  # 30 seconds at 30 fps
    while True:
        frame_no += 1
        print(f"Frame {frame_no}")
        in_bytes = reader.stdout.read(width * height * 3)
        if len(in_bytes) != width * height * 3:
            break

        in_array = (
            np.frombuffer(
                in_bytes,
                dtype=np.uint8,
            )
            .astype(np.float32)
            .reshape(shape)
        )

        if average_count < average_max:
            average_count += 1
            average = add_average(average, in_array, average_count)
        else:
            break

avg_image = Image.fromarray(average.astype(np.uint8), mode="RGB")
avg_image.save("experimental/40_bg_avg/avg.png")
