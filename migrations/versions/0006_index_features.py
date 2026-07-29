"""Persist document indexing features in the catalog."""

from alembic import op
import sqlalchemy as sa

revision = "0006_index_features"
down_revision = "0005_document_assets"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "document_catalog",
        sa.Column(
            "index_strategy",
            sa.String(length=32),
            nullable=False,
            server_default="auto",
        ),
    )
    op.add_column(
        "document_catalog",
        sa.Column(
            "use_raptor",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
    )
    op.add_column(
        "document_catalog",
        sa.Column(
            "use_graphrag",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
    )
    op.execute(
        """
        UPDATE document_catalog AS document
        SET
            index_strategy = latest.strategy,
            use_raptor = latest.use_raptor,
            use_graphrag = latest.use_graphrag
        FROM (
            SELECT DISTINCT ON (doc_id)
                doc_id,
                strategy,
                use_raptor,
                use_graphrag
            FROM index_jobs
            WHERE doc_id IS NOT NULL
              AND status = 'completed'
            ORDER BY doc_id, updated_at DESC
        ) AS latest
        WHERE document.doc_id = latest.doc_id
        """
    )


def downgrade() -> None:
    op.drop_column("document_catalog", "use_graphrag")
    op.drop_column("document_catalog", "use_raptor")
    op.drop_column("document_catalog", "index_strategy")
