"""Add the message_attachment reference manifest table.

Phase: EXPAND

Stores per-session chat attachment *references* (attachment id, size, MIME
type and the associated message/version). The table never stores attachment
bytes or a storage path, and deleting an attachment only flips ``is_deleted``
so the historical marker survives as an inaccessible tombstone.

Revision ID: f8a1b2c3d4e5
Revises: e5a7c9b1d3f2
Create Date: 2026-11-06

"""

import sqlalchemy as sa
from alembic import op
from langflow.utils import migration

revision = "f8a1b2c3d4e5"  # pragma: allowlist secret -- migration revision identifier
down_revision = "e5a7c9b1d3f2"  # pragma: allowlist secret -- migration revision identifier
branch_labels = None
depends_on = None

_TABLE = "message_attachment"

_INDEXES = (
    ("ix_message_attachment_session_id", ["session_id"], False),
    ("ix_message_attachment_user_id", ["user_id"], False),
    ("ix_message_attachment_tenant_id", ["tenant_id"], False),
    ("ix_message_attachment_message_id", ["message_id"], False),
    ("ix_message_attachment_session_created", ["session_id", "created_at"], False),
    ("ix_message_attachment_flow_session", ["flow_id", "session_id"], False),
)


def upgrade() -> None:
    conn = op.get_bind()
    if not migration.table_exists(_TABLE, conn):
        op.create_table(
            _TABLE,
            sa.Column("id", sa.Uuid(), primary_key=True, nullable=False),
            sa.Column("session_id", sa.String(length=255), nullable=False),
            sa.Column("flow_id", sa.Uuid(), nullable=True),
            sa.Column("user_id", sa.Uuid(), nullable=True),
            sa.Column("tenant_id", sa.String(length=255), nullable=True),
            sa.Column("message_id", sa.Uuid(), nullable=True),
            sa.Column("message_version", sa.Integer(), nullable=False),
            sa.Column("size", sa.BigInteger(), nullable=False),
            sa.Column("mime_type", sa.String(length=255), nullable=False),
            sa.Column("is_deleted", sa.Boolean(), nullable=False),
            sa.Column("deleted_at", sa.DateTime(), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("updated_at", sa.DateTime(), nullable=False),
        )

    inspector = sa.inspect(conn)
    existing = {index["name"] for index in inspector.get_indexes(_TABLE)}
    for name, columns, unique in _INDEXES:
        if name not in existing:
            op.create_index(name, _TABLE, columns, unique=unique)


def downgrade() -> None:
    conn = op.get_bind()
    if not migration.table_exists(_TABLE, conn):
        return
    inspector = sa.inspect(conn)
    existing = {index["name"] for index in inspector.get_indexes(_TABLE)}
    for name, _columns, _unique in reversed(_INDEXES):
        if name in existing:
            op.drop_index(name, table_name=_TABLE)
    op.drop_table(_TABLE)
