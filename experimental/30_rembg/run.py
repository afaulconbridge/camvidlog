# pip install rembg
import ffmpeg
from PIL import Image
from rembg import new_session, remove

ffmpeg_src = "data/20231117214546_VD_00327.MP4"
ffmpeg_dst = ffmpeg_src[:-4] + ".rembg.mp4"

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
    # session = new_session()
    session = new_session("birefnet-massive")

    frame_no = -1
    while True:
        frame_no += 1
        print(f"Frame {frame_no}")
        in_bytes = reader.stdout.read(width * height * 3)
        if len(in_bytes) != width * height * 3:
            break
        in_image = Image.frombytes("RGB", (width, height), in_bytes).convert("RGB")
        out_image = remove(in_image, session=session, post_process_mask=True)
        out_image = out_image.convert("RGB")
        out_bytes = out_image.tobytes()
        writer.stdin.write(out_bytes)
writer.stdin.close()
writer.wait()
