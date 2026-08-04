"""Add reversible retrieval availability to indexed documents."""

from alembic import op
import sqlalchemy as sa

revision = "0009_document_enabled"
down_revision = "0008_history_trace_id"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "document_catalog",
        sa.Column(
            "enabled",
            sa.Boolean(),
            nullable=False,
            server_default=sa.true(),
        ),
    )


def downgrade() -> None:
    op.drop_column("document_catalog", "enabled")
