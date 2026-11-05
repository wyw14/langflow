"""Data access for the chat attachment reference manifest."""

from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import update
from sqlmodel import col, select
from sqlmodel.ext.asyncio.session import AsyncSession

from langflow.services.database.models.message.model import MessageTable
from langflow.services.database.models.message_attachment.model import (
    MessageAttachment,
    MessageAttachmentCreate,
)


async def create_message_attachment(
    session: AsyncSession,
    *,
    data: MessageAttachmentCreate,
    session_id: str,
    user_id: UUID,
    flow_id: UUID | None,
    tenant_id: str | None,
) -> MessageAttachment:
    """Persist a newly registered attachment reference.

    ``user_id``/``flow_id``/``tenant_id`` are stamped server-side from the
    authorized session context; a caller cannot register a reference under
    another user's or tenant's scope.
    """
    # Build kwargs without ``id`` when the caller omitted it so the model's
    # ``default_factory`` assigns a new UUID (an explicit None fails UUID
    # validation on the non-optional primary key field).
    attachment_kwargs = {
        "session_id": session_id,
        "flow_id": flow_id,
        "user_id": user_id,
        "tenant_id": tenant_id,
        "message_id": data.message_id,
        "message_version": data.message_version,
        "size": data.size,
        "mime_type": data.mime_type,
    }
    if data.id is not None:
        attachment_kwargs["id"] = data.id
    attachment = MessageAttachment(**attachment_kwargs)
    session.add(attachment)
    await session.flush()
    await session.refresh(attachment)
    return attachment


async def get_message_attachment_for_user(
    session: AsyncSession,
    user_id: UUID,
    attachment_id: UUID,
) -> MessageAttachment | None:
    """Fetch one reference constrained to the owner; foreign IDs resolve to None."""
    stmt = select(MessageAttachment).where(
        MessageAttachment.id == attachment_id,
        col(MessageAttachment.user_id) == user_id,
    )
    return (await session.exec(stmt)).first()


async def list_message_attachments_for_user_session(
    session: AsyncSession,
    *,
    user_id: UUID,
    session_id: str,
    tenant_id: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> list[MessageAttachment]:
    """List a session's references for the owner, tombstoned rows included."""
    stmt = select(MessageAttachment).where(
        col(MessageAttachment.user_id) == user_id,
        MessageAttachment.session_id == session_id,
    )
    if tenant_id is not None:
        stmt = stmt.where(col(MessageAttachment.tenant_id) == tenant_id)
    stmt = stmt.order_by(
        col(MessageAttachment.created_at).desc(),
        col(MessageAttachment.id).desc(),
    )
    stmt = stmt.offset(offset).limit(limit)
    return list((await session.exec(stmt)).all())


async def soft_delete_message_attachment(attachment: MessageAttachment) -> MessageAttachment:
    """Turn a reference into an inaccessible tombstone instead of removing it.

    Idempotent: an already tombstoned reference keeps its original
    ``deleted_at`` and stays in the manifest.
    """
    if not attachment.is_deleted:
        attachment.is_deleted = True
        attachment.deleted_at = datetime.now(timezone.utc)
        attachment.updated_at = datetime.now(timezone.utc)
    return attachment


async def bump_message_attachment_versions(
    session: AsyncSession,
    message_id: UUID,
) -> int:
    """Advance the associated message version on every live reference.

    Called when the message is edited. Tombstoned references are left at the
    version at which their attachment was deleted.
    """
    result = await session.exec(
        update(MessageAttachment)
        .where(
            col(MessageAttachment.message_id) == message_id,
            col(MessageAttachment.is_deleted).is_(False),
        )
        .values(message_version=MessageAttachment.message_version + 1, updated_at=datetime.now(timezone.utc))
    )
    return result.rowcount or 0


async def get_session_tenant_ids(
    session: AsyncSession,
    session_id: str,
    *,
    user_id: UUID | None = None,
) -> set[str]:
    """Distinct tenant IDs recorded on a session's messages.

    Tenant context lives in ``MessageTable.session_metadata['tenant_id']``.
    Extraction runs in Python so the read is identical on SQLite and PostgreSQL
    (the JSON accessor syntax differs across dialects). Only non-empty strings
    count as a tenant claim.
    """
    stmt = select(MessageTable.session_metadata).where(MessageTable.session_id == session_id)
    if user_id is not None:
        stmt = stmt.where(col(MessageTable.user_id) == user_id)
    tenants: set[str] = set()
    for metadata in await session.exec(stmt):
        if isinstance(metadata, dict):
            tenant = metadata.get("tenant_id")
            if isinstance(tenant, str) and tenant.strip():
                tenants.add(tenant.strip())
    return tenants


async def get_stamped_session_tenant_ids(
    session: AsyncSession,
    *,
    user_id: UUID,
    session_id: str,
) -> set[str]:
    """Tenant IDs stamped on a session's reference rows.

    Falls back for tenant checks after the session's messages have been purged.
    """
    stmt = select(MessageAttachment.tenant_id).where(
        col(MessageAttachment.user_id) == user_id,
        MessageAttachment.session_id == session_id,
        col(MessageAttachment.tenant_id).isnot(None),
    )
    return {tenant for tenant in await session.exec(stmt) if isinstance(tenant, str) and tenant.strip()}


async def session_has_owned_attachments(
    session: AsyncSession,
    *,
    user_id: UUID,
    session_id: str,
) -> bool:
    """Whether the owner still has reference rows for a session whose messages are gone."""
    stmt = select(MessageAttachment.id).where(
        col(MessageAttachment.user_id) == user_id,
        MessageAttachment.session_id == session_id,
    )
    return (await session.exec(stmt)).first() is not None
