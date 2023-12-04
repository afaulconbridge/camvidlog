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

        tracks = cv_service.find_things(video_path)
        results = cv_service.know_tracks(tracks)
        print(filename)
        for _,p in results.items():
            print(p)

    # db_service = DbService(config.database_url)
    # db_service.add_video(filename=args.filename)
