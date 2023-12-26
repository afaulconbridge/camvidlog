import argparse
import logging
from io import BytesIO
from pathlib import Path

from numpy import ndarray
from PIL import Image, ImageOps

from camvidlog.config import ConfigService
from camvidlog.cv.service import ComputerVisionService
from camvidlog.db.service import DbService

logger = logging.getLogger(__name__)


def make_thumbnail(source: ndarray, size=(120, 120)) -> bytes:
    image = Image.fromarray(source)
    image.thumbnail(size)
    image = ImageOps.pad(image, size)
    with BytesIO() as tmpfile:
        image.save(tmpfile, format="jpeg")
        return tmpfile.getvalue()


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

        tracks = cv_service.find_things(video_path)
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
