"""Chat attachment reference manifest (``message_attachment`` table).

Each row is a *reference* to an attachment used by a chat message, grouped by
``session_id``. It deliberately stores manifest metadata only — attachment ID,
size, MIME type, and the associated message/version — and never stores file
bytes or a storage path, so the manifest cannot become a channel for content or
storage-location disclosure.

When an attachment is deleted the row is retained as a tombstone
(``is_deleted=True`` / ``accessible=False``): historical turns keep showing that
an attachment existed while the marker is reported as inaccessible. There is no
foreign key to ``message`` because the reference must survive message/session
cleanup just as it survives attachment deletion.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from uuid import UUID, uuid4

import sqlalchemy as sa
from pydantic import field_validator, model_validator
from sqlalchemy import Index
from sqlmodel import Column, Field, SQLModel

# RFC 6838 "name = token "/" token" shape (without the obsolete x- / wild-card
# leniency). Bounded length keeps the column and index portable across stores.
_MIME_TYPE_RE = re.compile(r"^[A-Za-z0-9!#$&^_.+-]+/[A-Za-z0-9!#$&^_.+-]+$")
_MAX_MIME_LENGTH = 255


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class MessageAttachmentBase(SQLModel):
    session_id: str = Field(sa_column=Column(sa.String(255), nullable=False))
    # Owning flow/user stamped from the message the reference belongs to.
    # Nullable for legacy/edge contexts, mirroring ``MessageTable``.
    flow_id: UUID | None = Field(default=None)
    user_id: UUID | None = Field(default=None)
    # Tenant copied from the message's ``session_metadata['tenant_id']`` so the
    # manifest can be authorized independently of the message row.
    tenant_id: str | None = Field(default=None, sa_column=Column(sa.String(255), nullable=True))
    # Plain UUID (no FK): the marker outlives message deletion.
    message_id: UUID | None = Field(default=None)
    # Version of the associated message this reference belongs to (1 for the
    # original turn, incremented when the message is edited).
    message_version: int = Field(default=1, sa_column=Column(sa.Integer(), nullable=False))
    size: int = Field(sa_column=Column(sa.BigInteger(), nullable=False))
    mime_type: str = Field(sa_column=Column(sa.String(_MAX_MIME_LENGTH), nullable=False))


class MessageAttachment(MessageAttachmentBase, table=True):  # type: ignore[call-arg]
    __tablename__ = "message_attachment"
    __table_args__ = (
        Index("ix_message_attachment_session_id", "session_id"),
        Index("ix_message_attachment_user_id", "user_id"),
        Index("ix_message_attachment_tenant_id", "tenant_id"),
        Index("ix_message_attachment_message_id", "message_id"),
        # Manifest reads filter by session and page newest-first; covering the
        # session with created_at keeps that walk index-backed.
        Index("ix_message_attachment_session_created", "session_id", "created_at"),
        Index("ix_message_attachment_flow_session", "flow_id", "session_id"),
    )

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    # Tombstone: a deleted attachment keeps its row (history stays intact) but
    # is reported as an inaccessible marker instead of a usable attachment.
    is_deleted: bool = Field(default=False, sa_column=Column(sa.Boolean(), nullable=False))
    deleted_at: datetime | None = Field(default=None)
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)

    @field_validator("size", mode="before")
    @classmethod
    def validate_size(cls, value):
        if value is None or int(value) < 0:
            msg = "Attachment size must be a non-negative integer."
            raise ValueError(msg)
        return int(value)

    @field_validator("mime_type")
    @classmethod
    def validate_mime_type(cls, value: str) -> str:
        if not value or len(value) > _MAX_MIME_LENGTH or not _MIME_TYPE_RE.fullmatch(value.strip()):
            msg = "Invalid MIME type; expected 'type/subtype'."
            raise ValueError(msg)
        return value.strip()

    @field_validator("flow_id", "user_id", "message_id", mode="before")
    @classmethod
    def validate_uuid(cls, value):
        if value is None or isinstance(value, UUID):
            return value
        return UUID(str(value))


class MessageAttachmentRead(MessageAttachmentBase):
    """Manifest wire shape — metadata only, never bytes or a storage path."""

    id: UUID
    is_deleted: bool = False
    # Always derived from the tombstone flag so a deleted marker can never be
    # reported as accessible.
    accessible: bool = True
    deleted_at: datetime | None = None
    created_at: datetime
    updated_at: datetime

    @model_validator(mode="after")
    def derive_accessible(self) -> MessageAttachmentRead:
        self.accessible = not self.is_deleted
        return self


class MessageAttachmentCreate(SQLModel):
    """Registration payload for one attachment reference in a session."""

    # Caller may pin the reference to the uploaded file's ID; otherwise one is
    # generated.
    id: UUID | None = None
    size: int = Field(ge=0)
    mime_type: str = Field(min_length=3, max_length=_MAX_MIME_LENGTH)
    message_id: UUID | None = None
    message_version: int = Field(default=1, ge=1)
    flow_id: UUID | None = None

    @field_validator("mime_type")
    @classmethod
    def validate_mime_type(cls, value: str) -> str:
        if not value or not _MIME_TYPE_RE.fullmatch(value.strip()):
            msg = "Invalid MIME type; expected 'type/subtype'."
            raise ValueError(msg)
        return value.strip()

    @field_validator("id", "message_id", "flow_id", mode="before")
    @classmethod
    def validate_uuids(cls, value):
        if value is None or isinstance(value, UUID):
            return value
        return UUID(str(value))
