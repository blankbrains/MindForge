"""Track the configuration used to build each document index."""

from alembic import op
import sqlalchemy as sa

revision = "0004_document_index_signature"
down_revision = "0003_index_jobs"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "document_catalog",
        sa.Column(
            "index_signature",
            sa.String(length=64),
            nullable=True,
        ),
    )


def downgrade() -> None:
    op.drop_column("document_catalog", "index_signature")
