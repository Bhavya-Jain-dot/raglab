"""Add workspaces, versioned datasets, vector cache, credentials and audit data.

The first project migration used metadata.create_all, so this migration is
deliberately idempotent for both databases that already ran 0001 and fresh
databases where 0001 created the current metadata.
"""
from alembic import op
from sqlalchemy import Boolean, Column, Integer, JSON, String, Text, inspect, select, text

from app.database import Base
from app import models  # noqa: F401

revision = "0002_product_foundation"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def _add_missing_columns(connection):
    additions = {
        "projects": [Column("workspace_id", Integer, nullable=True)],
        "datasets": [Column("description", Text, nullable=False, server_default="")],
        "examples": [
            Column("reference_contexts", JSON, nullable=False, server_default=text("'[]'")),
            Column("split", String(20), nullable=False, server_default="evaluation"),
        ],
        "documents": [
            Column("content_hash", String(64), nullable=True),
            Column("storage_path", String(1024), nullable=True),
            Column("status", String(24), nullable=False, server_default="ready"),
            Column("metadata_json", JSON, nullable=False, server_default=text("'{}'")),
        ],
        "experiments": [
            Column("owner_id", Integer, nullable=True),
            Column("cancel_requested", Boolean, nullable=False, server_default=text("false")),
            Column("attempt_count", Integer, nullable=False, server_default="0"),
            Column("max_attempts", Integer, nullable=False, server_default="3"),
            Column("progress_total", Integer, nullable=False, server_default="0"),
            Column("progress_completed", Integer, nullable=False, server_default="0"),
        ],
        "results": [
            Column("evaluation_details", JSON, nullable=False, server_default=text("'{}'")),
            Column("latency_data", JSON, nullable=False, server_default=text("'{}'")),
            Column("cost_data", JSON, nullable=False, server_default=text("'{}'")),
        ],
        "traces": [Column("external_trace_id", String(128), nullable=True)],
    }
    existing_tables = set(inspect(connection).get_table_names())
    for table_name, columns in additions.items():
        if table_name not in existing_tables:
            continue
        existing_columns = {column["name"] for column in inspect(connection).get_columns(table_name)}
        for column in columns:
            if column.name not in existing_columns:
                op.add_column(table_name, column)


def upgrade():
    connection = op.get_bind()
    if connection.dialect.name == "postgresql":
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    _add_missing_columns(connection)
    Base.metadata.create_all(bind=connection)

    # Put existing owner-scoped projects into a default workspace without
    # changing their access semantics.
    from app.models import Project, User, Workspace, WorkspaceMember

    users = connection.execute(select(User.id, User.name)).all()
    for user_id, name in users:
        workspace_id = connection.execute(
            select(Workspace.id).where(Workspace.owner_id == user_id).limit(1)
        ).scalar_one_or_none()
        if workspace_id is None:
            inserted = connection.execute(
                Workspace.__table__.insert().values(
                    name=f"{name or 'Personal'} workspace", owner_id=user_id
                )
            )
            workspace_id = inserted.inserted_primary_key[0]
            connection.execute(
                WorkspaceMember.__table__.insert().values(
                    workspace_id=workspace_id, user_id=user_id, role="owner"
                )
            )
        connection.execute(
            Project.__table__.update()
            .where(Project.owner_id == user_id, Project.workspace_id.is_(None))
            .values(workspace_id=workspace_id)
        )


def downgrade():
    # Keep user-created workspace data intact. The project has no destructive
    # downgrade path for this additive authorization/schema migration.
    pass
