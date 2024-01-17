# CamVidLog

[![PyPI - Version](https://img.shields.io/pypi/v/camvidlog.svg)](https://pypi.org/project/camvidlog)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/camvidlog.svg)](https://pypi.org/project/camvidlog)

-----

**Table of Contents**

- [Installation](#installation)
- [Execution](#execution)
- [Development](#development)
- [License](#license)

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


## License

`camvidlog` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.

## Blog

### Image enhancement

The primary source is a trail camera that has an infrared light. This tends to produce images that are well or over-lit in the centre, and under-lit at the edges. So the idea was that by enhancing the source image, the matches would be better. After all, this works for humans. 

In practice, this does not work. We can do nice contrast balancing, and it works well in thumbnail generation, but the image detection seems to become worse rather than better.

### Difficult animals

One of the more challenging animals seems to be hedgehogs. It might be because these are particularly small relative to the entire image. 20231120033202_VD_00336.MP4 is an example of a hedgehog. Foxes are also something that are often missing from AI training datasets and an example is data/20231121002042_VD_00342.MP4