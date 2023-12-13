import argparse
import logging
from pathlib import Path

from camvidlog.config import ConfigService
from camvidlog.cv.service import ComputerVisionService
from camvidlog.db.service import DbService

logger = logging.getLogger(__name__)

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
        video_path = Path(filename).resolve()

        tracks = cv_service.find_things(video_path)
        logger.info(f"Found {len(tracks)} tracks")

        db_service.add_video(filename=filename)

        for i, track in enumerate(tracks):
            logger.info(f"{i}) {track.frame_first}=>{track.frame_last}")
            db_service.add_track(filename=filename, frame_first=track.frame_first, frame_last=track.frame_last)
