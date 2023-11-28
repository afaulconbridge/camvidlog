import argparse
from pathlib import Path

from camvidlog.config import ConfigService
from camvidlog.cv import ComputerVisionService
from camvidlog.db.service import DbService

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="CamVidLogLoader", description="Loads files into CamVidLog")
    parser.add_argument("filename", nargs="+")
    args = parser.parse_args()

    config = ConfigService()
    cv_service = ComputerVisionService()

    for filename in args.filename:
        video_path = Path(filename).resolve()

        results = cv_service.analyse_video(video_path)
        print(results)

    # db_service = DbService(config.database_url)
    # db_service.add_video(filename=args.filename)
