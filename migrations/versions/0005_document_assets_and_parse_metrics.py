"""Persist parser assets and detailed job metrics."""

from alembic import op
import sqlalchemy as sa

revision = "0005_document_assets"
down_revision = "0004_document_index_signature"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "document_catalog",
        sa.Column(
            "parser_metadata",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'{}'"),
        ),
    )
    op.add_column(
        "index_jobs",
        sa.Column(
            "metrics",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'{}'"),
        ),
    )
    op.create_table(
        "document_assets",
        sa.Column("asset_id", sa.String(length=64), nullable=False),
        sa.Column("doc_id", sa.String(length=64), nullable=False),
        sa.Column("kind", sa.String(length=32), nullable=False),
        sa.Column("page", sa.Integer(), nullable=True),
        sa.Column("element_index", sa.Integer(), nullable=True),
        sa.Column("relative_path", sa.String(length=1024), nullable=True),
        sa.Column("content_type", sa.String(length=128), nullable=True),
        sa.Column("sha256", sa.String(length=64), nullable=True),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("width", sa.Integer(), nullable=True),
        sa.Column("height", sa.Integer(), nullable=True),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["doc_id"],
            ["document_catalog.doc_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("asset_id"),
    )
    op.create_index(
        op.f("ix_document_assets_doc_id"),
        "document_assets",
        ["doc_id"],
        unique=False,
    )
    op.create_index(
        "ix_document_assets_doc_page",
        "document_assets",
        ["doc_id", "page"],
        unique=False,
    )
    op.create_index(
        "ix_document_assets_doc_element",
        "document_assets",
        ["doc_id", "element_index"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_document_assets_doc_element", table_name="document_assets")
    op.drop_index("ix_document_assets_doc_page", table_name="document_assets")
    op.drop_index(op.f("ix_document_assets_doc_id"), table_name="document_assets")
    op.drop_table("document_assets")
    op.drop_column("index_jobs", "metrics")
    op.drop_column("document_catalog", "parser_metadata")
