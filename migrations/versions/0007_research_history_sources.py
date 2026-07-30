"""Persist citation sources with research history."""

from alembic import op
import sqlalchemy as sa

revision = "0007_history_sources"
down_revision = "0006_index_features"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "research_history",
        sa.Column("sources", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("research_history", "sources")
