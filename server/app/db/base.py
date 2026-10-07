from datetime import datetime
from typing import Annotated

from sqlalchemy import DateTime, MetaData
from sqlalchemy.orm import DeclarativeBase, mapped_column

metadata = MetaData(
    naming_convention={
        "ix": "ix_%(column_0_label)s",
        "uq": "uq_%(table_name)s_%(column_0_name)s",
        "ck": "ck_%(table_name)s_%(constraint_name)s",
        "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
        "pk": "pk_%(table_name)s",
    }
)

CreatedAt = Annotated[datetime, mapped_column(DateTime, nullable=False, default=datetime.now)]
UpdatedAt = Annotated[
    datetime,
    mapped_column(DateTime, nullable=False, default=datetime.now, onupdate=datetime.now),
]


class Base(DeclarativeBase):
    metadata = metadata
