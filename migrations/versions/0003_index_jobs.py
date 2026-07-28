"""Add persistent asynchronous indexing jobs."""

from alembic import op
import sqlalchemy as sa

revision = "0003_index_jobs"
down_revision = "0002_document_catalog"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "index_jobs",
        sa.Column("job_id", sa.String(length=64), nullable=False),
        sa.Column("doc_id", sa.String(length=64), nullable=True),
        sa.Column("filename", sa.String(length=512), nullable=False),
        sa.Column("file_path", sa.String(length=2048), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("stage", sa.String(length=32), nullable=False),
        sa.Column("progress", sa.Float(), nullable=False),
        sa.Column("chunk_count", sa.Integer(), nullable=False),
        sa.Column("timings", sa.JSON(), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("cancel_requested", sa.Boolean(), nullable=False),
        sa.Column("strategy", sa.String(length=32), nullable=False),
        sa.Column("use_raptor", sa.Boolean(), nullable=False),
        sa.Column("use_graphrag", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("job_id"),
    )
    op.create_index(
        op.f("ix_index_jobs_doc_id"),
        "index_jobs",
        ["doc_id"],
        unique=False,
    )
    op.create_index(
        "ix_index_jobs_status_created",
        "index_jobs",
        ["status", "created_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_index_jobs_status_created",
        table_name="index_jobs",
    )
    op.drop_index(
        op.f("ix_index_jobs_doc_id"),
        table_name="index_jobs",
    )
    op.drop_table("index_jobs")
