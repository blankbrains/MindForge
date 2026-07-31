"""Associate research history entries with observability traces."""

from alembic import op
import sqlalchemy as sa

revision = "0008_history_trace_id"
down_revision = "0007_history_sources"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "research_history",
        sa.Column("trace_id", sa.String(length=32), nullable=True),
    )
    op.create_index(
        "ix_research_history_trace_id",
        "research_history",
        ["trace_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_research_history_trace_id",
        table_name="research_history",
    )
    op.drop_column("research_history", "trace_id")
