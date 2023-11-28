from typing import Optional

from sqlmodel import Field, SQLModel


class Video(SQLModel, table=True):
    id_: Optional[int] = Field(default=None, primary_key=True, alias="id")
    filename: str = Field(unique=True)
