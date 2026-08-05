"""Add observable conversations, context snapshots, artifacts, and memories."""

from alembic import op
import sqlalchemy as sa

revision = "0010_context_memory"
down_revision = "0009_document_enabled"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "conversations",
        sa.Column("conversation_id", sa.String(length=32), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("title", sa.String(length=200), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("context_mode", sa.String(length=16), nullable=False),
        sa.Column("next_sequence", sa.BigInteger(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.CheckConstraint(
            "status IN ('active', 'archived', 'deleted')",
            name="ck_conversations_status",
        ),
        sa.CheckConstraint(
            "context_mode IN ('auto', 'manual', 'disabled')",
            name="ck_conversations_context_mode",
        ),
        sa.CheckConstraint(
            "next_sequence >= 1",
            name="ck_conversations_sequence",
        ),
        sa.CheckConstraint("version >= 1", name="ck_conversations_version"),
        sa.PrimaryKeyConstraint("conversation_id"),
    )
    op.create_index(
        "ix_conversations_user_status_updated",
        "conversations",
        ["user_id", "status", "updated_at"],
    )
    op.create_table(
        "conversation_messages",
        sa.Column("message_id", sa.String(length=32), nullable=False),
        sa.Column("conversation_id", sa.String(length=32), nullable=False),
        sa.Column("run_id", sa.String(length=80), nullable=True),
        sa.Column("role", sa.String(length=16), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("sequence", sa.BigInteger(), nullable=False),
        sa.Column("include_in_context", sa.Boolean(), nullable=False),
        sa.Column("pinned", sa.Boolean(), nullable=False),
        sa.Column("context_scope", sa.String(length=16), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("edited_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["conversation_id"],
            ["conversations.conversation_id"],
            ondelete="CASCADE",
        ),
        sa.CheckConstraint(
            "role IN ('user', 'assistant', 'system_notice')",
            name="ck_conversation_messages_role",
        ),
        sa.CheckConstraint(
            "context_scope IN ('turn', 'conversation', 'user')",
            name="ck_conversation_messages_scope",
        ),
        sa.CheckConstraint(
            "sequence >= 1",
            name="ck_conversation_messages_sequence",
        ),
        sa.PrimaryKeyConstraint("message_id"),
        sa.UniqueConstraint(
            "conversation_id",
            "sequence",
            name="uq_conversation_messages_sequence",
        ),
    )
    op.create_index(
        "ix_conversation_messages_context",
        "conversation_messages",
        ["conversation_id", "include_in_context", "deleted_at", "sequence"],
    )
    op.create_index(
        "ix_conversation_messages_run",
        "conversation_messages",
        ["run_id"],
    )
    op.create_table(
        "conversation_summaries",
        sa.Column("summary_id", sa.String(length=32), nullable=False),
        sa.Column("conversation_id", sa.String(length=32), nullable=False),
        sa.Column("from_sequence", sa.BigInteger(), nullable=False),
        sa.Column("to_sequence", sa.BigInteger(), nullable=False),
        sa.Column("summary", sa.JSON(), nullable=False),
        sa.Column("source_message_ids", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["conversation_id"],
            ["conversations.conversation_id"],
            ondelete="CASCADE",
        ),
        sa.CheckConstraint(
            "status IN ('active', 'invalidated', 'superseded')",
            name="ck_conversation_summaries_status",
        ),
        sa.CheckConstraint(
            "from_sequence >= 1 AND to_sequence >= from_sequence",
            name="ck_conversation_summaries_range",
        ),
        sa.PrimaryKeyConstraint("summary_id"),
    )
    op.create_index(
        "ix_conversation_summaries_active",
        "conversation_summaries",
        ["conversation_id", "status", "to_sequence"],
    )
    op.create_table(
        "research_artifacts",
        sa.Column("artifact_id", sa.String(length=32), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("run_id", sa.String(length=80), nullable=False),
        sa.Column("conversation_id", sa.String(length=32), nullable=True),
        sa.Column("artifact_type", sa.String(length=32), nullable=False),
        sa.Column("title", sa.String(length=300), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("source_ids", sa.JSON(), nullable=False),
        sa.Column("quality_score", sa.Float(), nullable=True),
        sa.Column("grounding_status", sa.String(length=24), nullable=False),
        sa.Column("freshness_class", sa.String(length=24), nullable=False),
        sa.Column("valid_from", sa.DateTime(timezone=True), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("embedding_version", sa.String(length=40), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["conversation_id"],
            ["conversations.conversation_id"],
            ondelete="CASCADE",
        ),
        sa.CheckConstraint(
            "artifact_type IN ('subtask_finding', 'report_section', 'claim', "
            "'evidence', 'decision', 'limitation', 'citation_source')",
            name="ck_research_artifacts_type",
        ),
        sa.CheckConstraint(
            "grounding_status IN ('grounded', 'model_only', 'not_required')",
            name="ck_research_artifacts_grounding",
        ),
        sa.CheckConstraint(
            "freshness_class IN ('stable', 'time_sensitive', 'volatile')",
            name="ck_research_artifacts_freshness",
        ),
        sa.PrimaryKeyConstraint("artifact_id"),
    )
    op.create_index(
        "ix_research_artifacts_recall",
        "research_artifacts",
        ["conversation_id", "enabled", "deleted_at", "created_at"],
    )
    op.create_index(
        "ix_research_artifacts_run",
        "research_artifacts",
        ["run_id"],
    )
    op.create_table(
        "context_snapshots",
        sa.Column("snapshot_id", sa.String(length=32), nullable=False),
        sa.Column("run_id", sa.String(length=80), nullable=False),
        sa.Column("conversation_id", sa.String(length=32), nullable=True),
        sa.Column("query_message_id", sa.String(length=32), nullable=True),
        sa.Column("standalone_query", sa.Text(), nullable=False),
        sa.Column("context_fingerprint", sa.String(length=64), nullable=False),
        sa.Column("budget_tokens", sa.Integer(), nullable=False),
        sa.Column("used_tokens", sa.Integer(), nullable=False),
        sa.Column("policy_version", sa.String(length=40), nullable=False),
        sa.Column("embedding_version", sa.String(length=40), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["conversation_id"],
            ["conversations.conversation_id"],
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["query_message_id"],
            ["conversation_messages.message_id"],
            ondelete="SET NULL",
        ),
        sa.CheckConstraint(
            "budget_tokens >= 0 AND used_tokens >= 0 "
            "AND used_tokens <= budget_tokens",
            name="ck_context_snapshots_budget",
        ),
        sa.PrimaryKeyConstraint("snapshot_id"),
        sa.UniqueConstraint("run_id", name="uq_context_snapshots_run"),
    )
    op.create_index(
        "ix_context_snapshots_conversation",
        "context_snapshots",
        ["conversation_id", "created_at"],
    )
    op.create_index(
        "ix_context_snapshots_created_at",
        "context_snapshots",
        ["created_at"],
    )
    op.create_table(
        "context_snapshot_items",
        sa.Column("snapshot_item_id", sa.String(length=32), nullable=False),
        sa.Column("snapshot_id", sa.String(length=32), nullable=False),
        sa.Column("source_type", sa.String(length=24), nullable=False),
        sa.Column("source_id", sa.String(length=64), nullable=False),
        sa.Column("content_snapshot", sa.Text(), nullable=False),
        sa.Column("rank", sa.Integer(), nullable=False),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=False),
        sa.Column("selection_reason", sa.String(length=200), nullable=False),
        sa.Column("pinned", sa.Boolean(), nullable=False),
        sa.Column("freshness_status", sa.String(length=16), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.ForeignKeyConstraint(
            ["snapshot_id"],
            ["context_snapshots.snapshot_id"],
            ondelete="CASCADE",
        ),
        sa.CheckConstraint(
            "rank >= 1",
            name="ck_context_snapshot_items_rank",
        ),
        sa.CheckConstraint(
            "token_count >= 0",
            name="ck_context_snapshot_items_tokens",
        ),
        sa.CheckConstraint(
            "freshness_status IN ('current', 'stale', 'expired')",
            name="ck_context_snapshot_items_freshness",
        ),
        sa.PrimaryKeyConstraint("snapshot_item_id"),
        sa.UniqueConstraint(
            "snapshot_id",
            "rank",
            name="uq_context_snapshot_items_rank",
        ),
    )
    op.create_index(
        "ix_context_snapshot_items_source",
        "context_snapshot_items",
        ["source_type", "source_id"],
    )
    op.create_table(
        "user_memories",
        sa.Column("memory_id", sa.String(length=32), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("category", sa.String(length=24), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("source_message_ids", sa.JSON(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("user_confirmed", sa.Boolean(), nullable=False),
        sa.Column("valid_from", sa.DateTime(timezone=True), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.CheckConstraint(
            "category IN ('preference', 'profile', 'stable_fact', "
            "'project_context', 'decision')",
            name="ck_user_memories_category",
        ),
        sa.CheckConstraint(
            "status IN ('active', 'superseded', 'forgotten')",
            name="ck_user_memories_status",
        ),
        sa.CheckConstraint(
            "confidence >= 0 AND confidence <= 1",
            name="ck_user_memories_confidence",
        ),
        sa.PrimaryKeyConstraint("memory_id"),
    )
    op.create_index(
        "ix_user_memories_active",
        "user_memories",
        ["user_id", "status", "deleted_at", "updated_at"],
    )
    op.create_table(
        "context_lineage",
        sa.Column("lineage_id", sa.String(length=32), nullable=False),
        sa.Column("source_type", sa.String(length=24), nullable=False),
        sa.Column("source_id", sa.String(length=64), nullable=False),
        sa.Column("derived_type", sa.String(length=24), nullable=False),
        sa.Column("derived_id", sa.String(length=64), nullable=False),
        sa.Column("relation", sa.String(length=32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "source_type IN ('message', 'artifact', 'memory')",
            name="ck_context_lineage_source_type",
        ),
        sa.CheckConstraint(
            "derived_type IN ('summary', 'artifact', 'memory', 'cache')",
            name="ck_context_lineage_derived_type",
        ),
        sa.PrimaryKeyConstraint("lineage_id"),
        sa.UniqueConstraint(
            "source_type",
            "source_id",
            "derived_type",
            "derived_id",
            "relation",
            name="uq_context_lineage_edge",
        ),
    )
    op.create_index(
        "ix_context_lineage_source",
        "context_lineage",
        ["source_type", "source_id"],
    )
    op.create_index(
        "ix_context_lineage_derived",
        "context_lineage",
        ["derived_type", "derived_id"],
    )
    op.create_table(
        "deletion_jobs",
        sa.Column("deletion_job_id", sa.String(length=32), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("target_type", sa.String(length=24), nullable=False),
        sa.Column("target_id", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.CheckConstraint(
            "status IN ('pending', 'running', 'completed', 'failed')",
            name="ck_deletion_jobs_status",
        ),
        sa.PrimaryKeyConstraint("deletion_job_id"),
    )
    op.create_index(
        "ix_deletion_jobs_user_created",
        "deletion_jobs",
        ["user_id", "created_at"],
    )
    op.add_column(
        "research_history",
        sa.Column("conversation_id", sa.String(length=32), nullable=True),
    )
    op.add_column(
        "research_history",
        sa.Column("run_id", sa.String(length=80), nullable=True),
    )
    op.create_index(
        "ix_research_history_conversation_id",
        "research_history",
        ["conversation_id"],
    )
    op.create_index(
        "ix_research_history_run_id",
        "research_history",
        ["run_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_research_history_run_id", table_name="research_history")
    op.drop_index(
        "ix_research_history_conversation_id",
        table_name="research_history",
    )
    op.drop_column("research_history", "run_id")
    op.drop_column("research_history", "conversation_id")
    op.drop_index("ix_deletion_jobs_user_created", table_name="deletion_jobs")
    op.drop_table("deletion_jobs")
    op.drop_index("ix_context_lineage_derived", table_name="context_lineage")
    op.drop_index("ix_context_lineage_source", table_name="context_lineage")
    op.drop_table("context_lineage")
    op.drop_index("ix_user_memories_active", table_name="user_memories")
    op.drop_table("user_memories")
    op.drop_index(
        "ix_context_snapshot_items_source",
        table_name="context_snapshot_items",
    )
    op.drop_table("context_snapshot_items")
    op.drop_index(
        "ix_context_snapshots_created_at",
        table_name="context_snapshots",
    )
    op.drop_index(
        "ix_context_snapshots_conversation",
        table_name="context_snapshots",
    )
    op.drop_table("context_snapshots")
    op.drop_index("ix_research_artifacts_run", table_name="research_artifacts")
    op.drop_index("ix_research_artifacts_recall", table_name="research_artifacts")
    op.drop_table("research_artifacts")
    op.drop_index(
        "ix_conversation_summaries_active",
        table_name="conversation_summaries",
    )
    op.drop_table("conversation_summaries")
    op.drop_index(
        "ix_conversation_messages_run",
        table_name="conversation_messages",
    )
    op.drop_index(
        "ix_conversation_messages_context",
        table_name="conversation_messages",
    )
    op.drop_table("conversation_messages")
    op.drop_index(
        "ix_conversations_user_status_updated",
        table_name="conversations",
    )
    op.drop_table("conversations")
