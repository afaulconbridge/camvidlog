from typing import Iterable, Optional

from sqlalchemy import Engine, create_engine
from sqlmodel import Session, SQLModel, select

from camvidlog.db.models import Track, Video


class DbService:
    db_url: str
    _engine: Engine

    def __init__(self, db_url: str):
        self.db_url = db_url

        # turn off forced logging via echo
        self._engine = create_engine(self.db_url, echo=False)

        SQLModel.metadata.create_all(self._engine)

    def add_video(self, filename: str) -> None:
        video = Video(filename=filename)
        with Session(self._engine) as session:
            session.add(video)
            session.commit()

    def add_track(
        self,
        filename: str,
        frame_first: int,
        frame_last: int,
        thumb_first: bytes,
        thumb_mid: bytes,
        thumb_last: bytes,
        result: str,
        result_score: float,
    ) -> None:
        with Session(self._engine) as session:
            video = session.exec(select(Video).where(Video.filename == filename)).one()
            track = Track(
                video_id=video.id_,
                frame_first=frame_first,
                frame_last=frame_last,
                thumb_first=thumb_first,
                thumb_mid=thumb_mid,
                thumb_last=thumb_last,
                result=result,
                result_score=result_score,
            )
            session.add(track)
            session.commit()

    def get_videos(self) -> Iterable[Video]:
        with Session(self._engine) as session:
            results = session.exec(select(Video))
            return results.fetchall()

    def get_tracks_with_videos(self, filter_results: Optional[Iterable[str]] = None) -> Iterable[tuple[Video, Track]]:
        with Session(self._engine) as session:
            query = select(Video, Track).join_from(Video, Track, isouter=True).order_by(Video.id_, Track.id_)
            if filter_results is not None:  # need to separate empty list or list of None
                if None in filter_results:
                    query = query.where(Track.result.in_(filter_results) | (Track.result == None))
                else:
                    query = query.where(Track.result.in_(filter_results))
            results = session.exec(query)
            return results.fetchall()

    def get_track_results(self) -> Iterable[str]:
        with Session(self._engine) as session:
            return session.exec(select(Track.result).distinct().order_by(Track.result)).fetchall()
