"""Chat attachment reference manifest endpoints.

Covers the session-scoped manifest (id/size/MIME/associated message version,
never body or storage path), tombstone retention after deletion, tenant
validation, and cross-user isolation.
"""

from uuid import UUID, uuid4

import pytest
from httpx import AsyncClient
from langflow.services.auth.utils import get_auth_service
from langflow.services.database.models.flow.model import Flow
from langflow.services.database.models.message.model import MessageTable
from langflow.services.database.models.message_attachment import (
    MessageAttachment,
    MessageAttachmentCreate,
    MessageAttachmentRead,
)
from langflow.services.database.models.user.model import User, UserRead
from langflow.services.deps import session_scope

BASE = "api/v1/monitor"


@pytest.fixture
async def other_active_user(client):  # noqa: ARG001
    username = f"other-attachment-user-{uuid4().hex[:8]}"
    async with session_scope() as session:
        user = User(
            username=username,
            password=get_auth_service().get_password_hash("testpassword"),
            is_active=True,
            is_superuser=False,
        )
        session.add(user)
        await session.flush()
        await session.refresh(user)
        user = UserRead.model_validate(user, from_attributes=True)

    yield user

    async with session_scope() as session:
        db_user = await session.get(User, user.id)
        if db_user:
            await session.delete(db_user)


@pytest.fixture
async def other_logged_in_headers(client: AsyncClient, other_active_user):
    response = await client.post(
        "api/v1/login",
        data={"username": other_active_user.username, "password": "testpassword"},
    )
    assert response.status_code == 200
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


@pytest.fixture
async def chat_session(active_user):
    """Create an owned flow + message; returns a factory for fresh sessions."""

    async def _make(*, session_id: str | None = None, tenant_id: str | None = None):
        session_id = session_id or f"attach-session-{uuid4().hex[:10]}"
        async with session_scope() as db:
            flow = Flow(name=f"attach-flow-{uuid4().hex[:8]}", user_id=active_user.id, data={})
            db.add(flow)
            await db.flush()
            message = MessageTable(
                text="turn with attachment",
                sender="User",
                sender_name="User",
                session_id=session_id,
                flow_id=flow.id,
                user_id=active_user.id,
                category="message",
                files=[],
                properties={},
                content_blocks=[],
                session_metadata={"tenant_id": tenant_id} if tenant_id else None,
            )
            db.add(message)
            await db.flush()
            return {
                "session_id": session_id,
                "flow_id": flow.id,
                "message_id": message.id,
            }

    return _make


def _payload(message_id=None, **overrides):
    payload = {"size": 2048, "mime_type": "application/pdf"}
    if message_id is not None:
        payload["message_id"] = str(message_id)
    payload.update(overrides)
    return payload


async def _register(client, headers, session_id, *, tenant_id=None, **overrides):
    params = {"tenant_id": tenant_id} if tenant_id is not None else None
    return await client.post(
        f"{BASE}/messages/session/{session_id}/attachments",
        headers=headers,
        params=params,
        json=_payload(**overrides),
    )


async def _list(client, headers, session_id, **params):
    return await client.get(
        f"{BASE}/messages/session/{session_id}/attachments",
        headers=headers,
        params=params or None,
    )


async def _delete(client, headers, attachment_id, *, tenant_id=None):
    params = {"tenant_id": tenant_id} if tenant_id is not None else None
    return await client.delete(
        f"{BASE}/messages/attachments/{attachment_id}",
        headers=headers,
        params=params,
    )


def test_manifest_schema_has_no_body_or_path_fields():
    """The wire/table shape structurally cannot leak content or storage paths."""
    forbidden = {"path", "file_path", "storage_path", "content", "body", "data", "url"}
    read_fields = set(MessageAttachmentRead.model_fields)
    table_cols = set(MessageAttachment.model_fields)
    assert read_fields & forbidden == set()
    assert table_cols & forbidden == set()
    assert {"id", "size", "mime_type", "message_version", "session_id"} <= read_fields


def test_create_schema_validates_mime_and_size():
    attachment = MessageAttachmentCreate(size=10, mime_type="image/png")
    assert attachment.mime_type == "image/png"
    with pytest.raises(ValueError, match="MIME"):
        MessageAttachmentCreate(size=10, mime_type="not-a-mime")
    with pytest.raises(ValueError, match="greater than or equal to 0"):
        MessageAttachmentCreate(size=-1, mime_type="image/png")


async def test_register_and_list_returns_metadata_only(client, logged_in_headers, chat_session):
    target = await chat_session()

    response = await _register(
        client,
        logged_in_headers,
        target["session_id"],
        message_id=target["message_id"],
        size=4096,
        mime_type="image/png",
    )
    assert response.status_code == 201, response.text
    created = response.json()
    assert created["size"] == 4096
    assert created["mime_type"] == "image/png"
    assert created["message_id"] == str(target["message_id"])
    assert created["message_version"] == 1
    assert created["accessible"] is True
    assert created["is_deleted"] is False

    listed = await _list(client, logged_in_headers, target["session_id"])
    assert listed.status_code == 200, listed.text
    items = listed.json()
    assert len(items) == 1
    item = items[0]
    assert item["id"] == created["id"]
    for secret_key in ("path", "file_path", "storage_path", "content", "body", "data", "url"):
        assert secret_key not in item


async def test_legacy_session_without_tenant_is_allowed(client, logged_in_headers, chat_session):
    target = await chat_session()
    response = await _register(client, logged_in_headers, target["session_id"])
    assert response.status_code == 201, response.text
    assert response.json()["tenant_id"] is None

    listed = await _list(client, logged_in_headers, target["session_id"])
    assert listed.status_code == 200


async def test_claimed_tenant_requires_matching_context(client, logged_in_headers, chat_session):
    target = await chat_session(tenant_id="tenant-acme")

    # Missing context -> 403 on register and list.
    missing = await _register(client, logged_in_headers, target["session_id"])
    assert missing.status_code == 403
    missing_list = await _list(client, logged_in_headers, target["session_id"])
    assert missing_list.status_code == 403

    # Wrong tenant -> 403.
    wrong = await _register(client, logged_in_headers, target["session_id"], tenant_id="tenant-globex")
    assert wrong.status_code == 403

    # Matching tenant -> allowed and the reference is tenant-stamped.
    ok = await _register(client, logged_in_headers, target["session_id"], tenant_id="tenant-acme")
    assert ok.status_code == 201, ok.text
    assert ok.json()["tenant_id"] == "tenant-acme"

    listed = await _list(client, logged_in_headers, target["session_id"], tenant_id="tenant-acme")
    assert listed.status_code == 200
    assert len(listed.json()) == 1


async def test_delete_retains_inaccessible_tombstone(client, logged_in_headers, chat_session):
    target = await chat_session()
    created = (await _register(client, logged_in_headers, target["session_id"])).json()

    deleted = await _delete(client, logged_in_headers, created["id"])
    assert deleted.status_code == 200, deleted.text
    marker = deleted.json()
    assert marker["id"] == created["id"]
    assert marker["accessible"] is False
    assert marker["is_deleted"] is True
    assert marker["deleted_at"] is not None
    # Metadata retained for the historical marker.
    assert marker["size"] == created["size"]
    assert marker["mime_type"] == created["mime_type"]

    # The row physically survives and is still listed, marked inaccessible.
    async with session_scope() as db:
        row = await db.get(MessageAttachment, UUID(created["id"]))
        assert row is not None
        assert row.is_deleted is True
    listed = (await _list(client, logged_in_headers, target["session_id"])).json()
    assert len(listed) == 1
    assert listed[0]["accessible"] is False

    # Idempotent: deleting again returns the same tombstone.
    again = await _delete(client, logged_in_headers, created["id"])
    assert again.status_code == 200
    assert again.json()["deleted_at"] == marker["deleted_at"]


async def test_tenant_guards_tombstone_delete(client, logged_in_headers, chat_session):
    target = await chat_session(tenant_id="tenant-acme")
    created = (await _register(client, logged_in_headers, target["session_id"], tenant_id="tenant-acme")).json()

    assert (await _delete(client, logged_in_headers, created["id"])).status_code == 403
    assert (await _delete(client, logged_in_headers, created["id"], tenant_id="tenant-globex")).status_code == 403
    ok = await _delete(client, logged_in_headers, created["id"], tenant_id="tenant-acme")
    assert ok.status_code == 200
    assert ok.json()["accessible"] is False


async def test_cross_user_isolation(client, logged_in_headers, other_logged_in_headers, chat_session):
    target = await chat_session()
    created = (await _register(client, logged_in_headers, target["session_id"])).json()

    # Another user cannot see the manifest, register into, or delete the session.
    foreign_list = await _list(client, other_logged_in_headers, target["session_id"])
    assert foreign_list.status_code == 404
    foreign_register = await _register(client, other_logged_in_headers, target["session_id"])
    assert foreign_register.status_code == 404
    foreign_delete = await _delete(client, other_logged_in_headers, created["id"])
    assert foreign_delete.status_code == 404

    # The owner's reference is untouched by the foreign delete attempt.
    async with session_scope() as db:
        row = await db.get(MessageAttachment, UUID(created["id"]))
        assert row.is_deleted is False


async def test_unknown_session_is_not_disclosed(client, logged_in_headers):
    unknown = f"missing-session-{uuid4().hex}"
    assert (await _list(client, logged_in_headers, unknown)).status_code == 404
    assert (await _register(client, logged_in_headers, unknown)).status_code == 404
    assert (await _delete(client, logged_in_headers, uuid4())).status_code == 404


async def test_requires_authentication(client, chat_session):
    # Missing credentials are refused (the app maps unauthenticated calls to 403).
    target = await chat_session()
    assert (await client.get(f"{BASE}/messages/session/{target['session_id']}/attachments")).status_code in (401, 403)


async def test_invalid_payload_rejected(client, logged_in_headers, chat_session):
    target = await chat_session()
    bad_mime = await client.post(
        f"{BASE}/messages/session/{target['session_id']}/attachments",
        headers=logged_in_headers,
        json={"size": 10, "mime_type": "filesystem/path/../../x"},
    )
    assert bad_mime.status_code == 422
    bad_size = await client.post(
        f"{BASE}/messages/session/{target['session_id']}/attachments",
        headers=logged_in_headers,
        json={"size": -5, "mime_type": "image/png"},
    )
    assert bad_size.status_code == 422


async def test_duplicate_attachment_id_conflicts(client, logged_in_headers, chat_session):
    target = await chat_session()
    attachment_id = uuid4()
    first = await _register(client, logged_in_headers, target["session_id"], id=str(attachment_id))
    assert first.status_code == 201, first.text
    second = await _register(client, logged_in_headers, target["session_id"], id=str(attachment_id))
    assert second.status_code == 409


async def test_message_edit_bumps_associated_version(client, logged_in_headers, chat_session):
    target = await chat_session()
    created = (
        await _register(
            client,
            logged_in_headers,
            target["session_id"],
            message_id=target["message_id"],
        )
    ).json()
    assert created["message_version"] == 1

    edited = await client.put(
        f"{BASE}/messages/{target['message_id']}",
        headers=logged_in_headers,
        json={"text": "edited turn"},
    )
    assert edited.status_code == 200, edited.text

    listed = (await _list(client, logged_in_headers, target["session_id"])).json()
    assert listed[0]["message_version"] == 2

    # A deleted (tombstoned) reference is pinned to its last version.
    await _delete(client, logged_in_headers, created["id"])
    await client.put(
        f"{BASE}/messages/{target['message_id']}",
        headers=logged_in_headers,
        json={"text": "edited again"},
    )
    listed = (await _list(client, logged_in_headers, target["session_id"])).json()
    assert listed[0]["message_version"] == 2
    assert listed[0]["accessible"] is False
