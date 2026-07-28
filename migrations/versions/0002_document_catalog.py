"""Add document catalog and research history query index."""

from alembic import op
import sqlalchemy as sa

revision = "0002_document_catalog"
down_revision = "0001_baseline"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_index(
        "ix_research_history_user_created_id",
        "research_history",
        ["user_id", "created_at", "id"],
        unique=False,
    )
    op.create_table(
        "document_catalog",
        sa.Column("doc_id", sa.String(length=64), nullable=False),
        sa.Column("filename", sa.String(length=512), nullable=False),
        sa.Column("chunk_count", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("doc_id"),
    )
    op.create_index(
        op.f("ix_document_catalog_status"),
        "document_catalog",
        ["status"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        op.f("ix_document_catalog_status"),
        table_name="document_catalog",
    )
    op.drop_table("document_catalog")
    op.drop_index(
        "ix_research_history_user_created_id",
        table_name="research_history",
    )
