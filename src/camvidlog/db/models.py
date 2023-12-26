from typing import Optional

from sqlmodel import Field, SQLModel


class Video(SQLModel, table=True):
    id_: Optional[int] = Field(default=None, primary_key=True)
    filename: str = Field(unique=True)


class Track(SQLModel, table=True):
    id_: Optional[int] = Field(default=None, primary_key=True)
    video_id: int = Field(default=None, foreign_key="video.id_")
    frame_first: int
    frame_last: int
    thumb_first: bytes
    thumb_mid: bytes
    thumb_last: bytes
