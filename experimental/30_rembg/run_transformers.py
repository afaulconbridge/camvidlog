# Load BiRefNet with weights
import ffmpeg
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

# ffmpeg_src = "data/20231117214546_VD_00327.MP4"
ffmpeg_src = "data/20231121005950_VD_00344.MP4"
ffmpeg_dst = ffmpeg_src[:-4] + ".mini.my_rmbg.mp4"


def extract_object(birefnet, image: Image):
    # Data settings
    image_size = (1024, 1024)
    transform_image = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    input_images = transform_image(image).unsqueeze(0).to("cuda")

    # Prediction
    with torch.no_grad():
        preds = birefnet(input_images)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    image_mask = pred_pil.resize(image.size)
    image_out = Image.composite(image, Image.new("RGB", image.size), image_mask)
    return image_out, image_mask


if __name__ == "__main__":
    birefnet = AutoModelForImageSegmentation.from_pretrained("ZhengPeng7/BiRefNet", trust_remote_code=True)

    torch.set_float32_matmul_precision(["high", "highest"][0])
    birefnet.to("cuda")
    birefnet.eval()

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

    frame_no = -1
    while True:
        frame_no += 1
        print(f"Frame {frame_no}")
        in_bytes = reader.stdout.read(width * height * 3)
        if len(in_bytes) != width * height * 3:
            break
        in_image = Image.frombytes("RGB", (width, height), in_bytes).convert("RGB")
        out_image, out_mask = extract_object(birefnet, in_image)
        out_image = out_image.convert("RGB")
        out_bytes = out_image.tobytes()
        writer.stdin.write(out_bytes)
writer.stdin.close()
writer.wait()
