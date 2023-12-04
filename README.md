# CamVidLog

[![PyPI - Version](https://img.shields.io/pypi/v/camvidlog.svg)](https://pypi.org/project/camvidlog)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/camvidlog.svg)](https://pypi.org/project/camvidlog)

-----

**Table of Contents**

- [Installation](#installation)
- [Execution](#execution)
- [License](#license)

## Installation

```console
pip install camvidlog
```

This requires several large (>100MB each) external data files:

wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-oiv7.pt -O src/camvidlog/data/yolov8x-oiv7.pt
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-cls.pt -O src/camvidlog/data/yolov8x-cls.pt

## Execution

Start the web application:

```console
hatch run shiny run -d src/camvidlog
```

Load a single video:

```console
hatch run python src/camvidlog/run/load.py FILENAME
```


## License

`camvidlog` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
