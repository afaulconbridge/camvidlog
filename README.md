# CamVidLog

[![PyPI - Version](https://img.shields.io/pypi/v/camvidlog.svg)](https://pypi.org/project/camvidlog)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/camvidlog.svg)](https://pypi.org/project/camvidlog)

-----

**Table of Contents**

- [CamVidLog](#camvidlog)
  - [Installation](#installation)
  - [Execution](#execution)
  - [Development](#development)
  - [License](#license)
  - [Blog](#blog)
    - [Image enhancement](#image-enhancement)
    - [Difficult animals](#difficult-animals)
  - [FFMPEG](#ffmpeg)

## Installation

```console
pip install camvidlog
```

This requires several large (>100MB each) external data files:

```console
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-oiv7.pt -O src/camvidlog/data/yolov8x-oiv7.pt
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-cls.pt -O src/camvidlog/data/yolov8x-cls.pt
```

## Execution

Start the web application:

```console
hatch run shiny run -d src/camvidlog
```

Load a single video:

```console
hatch run python src/camvidlog/run/load.py FILENAME
```

Load multiple videos:

```console
find data -name "*.MP4" -print0 | time xargs -0 hatch run python src/camvidlog/run/load.py
```

Note - in theory, should be able to do this in a single python process. However, PyTorch seems to run out of GPU memory if its run that way currently.

## Development

Run linting checks etc with:

```console
hatch -e lint run all
```

Monitor hardware with:

```console
nvtop
```

Note: when using a remote host dev container, connect to the host first then create the container from the repo. Also, recovery containers might not work so a command like this is helpful to get into the volume and edit if it won't start ("open in a recovery container" jumps back to the local host, which isn't useful as it doesn't contain the modified/broken files): `docker run -it --rm -v camvidlog-procs-d2e8b20ccf5ec88cb9c0dd5c615ddaadf78edc93fffe948d8a03d8ea302f07a8:/recovery debian`

## License

`camvidlog` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.

## Blog

### Image enhancement

The primary source is a trail camera that has an infrared light. This tends to produce images that are well or over-lit in the centre, and under-lit at the edges. So the idea was that by enhancing the source image, the matches would be better. After all, this works for humans.

In practice, this does not work. We can do nice contrast balancing, and it works well in thumbnail generation, but the image detection seems to become worse rather than better.

### Difficult animals

One of the more challenging animals seems to be hedgehogs. It might be because these are particularly small relative to the entire image. 20231120033202_VD_00336.MP4 is an example of a hedgehog. Foxes are also something that are often missing from AI training datasets and an example is data/20231121002042_VD_00342.MP4


## FFMPEG

ffmpeg -hide_banner -hwaccels
ffmpeg -hide_banner -encoders
ffmpeg -hide_banner -decoders
ffmpeg -hide_banner -hwaccel cuda -i input.mp4 output.mp4
ffmpeg -c:v h264_cuvid -i input.h264.mp4 output.h264.mp4
