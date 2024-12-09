from subprocess import Popen

import ffmpeg

if __name__ == "__main__":
    # ffmpeg -fflags nobuffer -flags low_delay -rtsp_transport tcp -i "rtsp://192.168.1.110:8554/cam1" -f image2 -vf fps=fps=1 hello/img%03d.png

    ffmpeg_kwargs = {
        "fflags": "nobuffer",
        "flags": "low_delay",
        "use_wallclock_as_timestamps": 1,
        "rtsp_transport": "tcp",
    }

    probe = ffmpeg.probe("rtsp://192.168.1.110:8554/cam1", **ffmpeg_kwargs)
    video_stream = next(
        (stream for stream in probe["streams"] if stream["codec_type"] == "video"),
        None,
    )
    width = int(video_stream["width"])
    height = int(video_stream["height"])

    reader: Popen = (
        ffmpeg.input(
            "rtsp://192.168.1.110:8554/cam1",
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
            # quiet=True,
        )
    )
    writer: Popen = (
        ffmpeg.input(
            "pipe:",
            format="rawvideo",
            pix_fmt="rgb24",
            hwaccel="cuda",
            s=f"{width}x{height}",
        )
        .output(
            "out.mp4",
            pix_fmt="yuv420p",
        )
        .overwrite_output()
        .run_async(
            pipe_stdin=True,
            # quiet=True,
        )
    )

    frame_no = -1
    while frame_no < 100:
        frame_no += 1
        print(f"Frame {frame_no}")
        in_bytes = reader.stdout.read(width * height * 3)
        if len(in_bytes) != width * height * 3:
            break

        out_bytes = in_bytes

        writer.stdin.write(out_bytes)

    print("closing")
    reader.kill()
    print("closed reader")
    writer.stdin.close()
    writer.wait()
    print("closed writer")
