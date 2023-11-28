from sqlalchemy import Engine, create_engine
from sqlmodel import Session, SQLModel

from camvidlog.db.models import Video


class DbService:
    db_url: str
    _engine: Engine

    def __init__(self, db_url: str):
        self.db_url = db_url

        self._engine = create_engine(self.db_url, echo=True)

        SQLModel.metadata.create_all(self._engine)

    def add_video(self, filename: str) -> None:
        video = Video(filename=filename)
        with Session(self._engine) as session:
            session.add(video)
            session.commit()
