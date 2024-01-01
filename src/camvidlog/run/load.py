import argparse
import logging
import math
from pathlib import Path

import cv2
import numpy as np
from numpy import ndarray

from camvidlog.config import ConfigService
from camvidlog.cv.service import ComputerVisionService
from camvidlog.db.service import DbService

logger = logging.getLogger(__name__)


def cv2_resize(image: ndarray, width: int, height: int, inter: int = cv2.INTER_AREA, border: int = 0):
    # based on https://github.com/PyImageSearch/imutils

    # initialize the dimensions of the image to be resized and
    # grab the image size
    (h, w) = image.shape[:2]
    if w > h:
        # calculate the ratio of the width and construct the dimensions
        r = width / w
        dim = (int(width), int(h * r))
    else:
        # calculate the ratio of the height and construct the dimensions
        r = height / h
        dim = (int(w * r), int(height))

    # resize the image
    # if making it smaller, use cv2.INTER_AREA
    # if making it larger, use cv2.INTER_CUBIC
    resized = cv2.resize(image, dim, interpolation=inter)

    # apply a border to get real size
    height_resized, width_resized, *_ = resized.shape
    border_left = math.floor((width - width_resized) / 2)
    border_right = math.ceil((width - width_resized) / 2)
    border_top = math.floor((height - height_resized) / 2)
    border_bottom = math.ceil((height - height_resized) / 2)
    if border_left or border_right or border_top or border_bottom:
        bordered = cv2.copyMakeBorder(
            resized, border_top, border_bottom, border_left, border_right, cv2.BORDER_CONSTANT, value=border
        )
        return bordered
    else:
        # no border needed
        return resized


def make_thumbnail(image: ndarray, size=(120, 120)) -> bytes:
    # need to be converted to grayscale before contrast manipulation
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # global histogram equalization
    image = cv2.equalizeHist(image)

    # local histogram equalization
    # clip = 1
    # tile = 4
    # clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    # image = clahe.apply(image)

    # resize and border
    # do this _after_ histogram normalization so that any border is not included in that step
    # if its making it smaller, use cv2.INTER_AREA
    # if its making it larger, use cv2.INTER_CUBIC
    if image.shape[0] < size[0] and image.shape[0] < size[1]:
        inter = cv2.INTER_AREA
    else:
        inter = cv2.INTER_CUBIC
    image = cv2_resize(image, size[0], size[1], inter)

    # turn image into jpeg bytes
    img_encode = cv2.imencode(".jpg", image)[1]
    data_encode = np.array(img_encode)
    byte_encode = data_encode.tobytes()

    return byte_encode


if __name__ == "__main__":
    # this should only happen once per python process
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(prog="CamVidLogLoader", description="Loads files into CamVidLog")
    parser.add_argument("filename", nargs="+")
    args = parser.parse_args()

    config = ConfigService()
    cv_service = ComputerVisionService()
    db_service = DbService(config.database_url)

    logger.info(f"Found {len(args.filename)} files")
    for filename in sorted(args.filename):
        # TODO move this into a separate VideoService ?

        video_path = Path(filename).resolve()

        tracks = cv_service.find_things(video_path, framestep=3)
        logger.info(f"Found {len(tracks)} tracks")

        db_service.add_video(filename=filename)

        for i, track in enumerate(tracks):
            logger.info(f"{i}) {track.frame_first}=>{track.frame_last}")

            # generate first/mid/last thumbnail images as JPEG files
            thumb_first = make_thumbnail(track.frames[0].sub_img)
            thumb_mid = make_thumbnail(track.frames[len(track.frames) // 2].sub_img)
            thumb_last = make_thumbnail(track.frames[-1].sub_img)

            # TODO more video metadata - date taken, FPS, etc

            db_service.add_track(
                filename=filename,
                frame_first=track.frame_first,
                frame_last=track.frame_last,
                thumb_first=thumb_first,
                thumb_mid=thumb_mid,
                thumb_last=thumb_last,
            )
