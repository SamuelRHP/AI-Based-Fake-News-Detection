"""Initial database schema: articles, predictions, feedbacks.

Revision ID: 001_initial_schema
Revises:
Create Date: 2026-10-12
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "001_initial_schema"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # 1. Articles table
    op.create_table(
        "articles",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("url", sa.String(length=2048), nullable=True),
        sa.Column("title", sa.String(length=512), nullable=True),
        sa.Column("raw_text", sa.Text(), nullable=False),
        sa.Column("cleaned_text", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_index(op.f("ix_articles_url"), "articles", ["url"], unique=False)

    # 2. Predictions table
    op.create_table(
        "predictions",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column(
            "article_id",
            sa.String(length=36),
            sa.ForeignKey("articles.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("verdict", sa.String(length=32), nullable=False),
        sa.Column("confidence_score", sa.Float(), nullable=False),
        sa.Column("prob_fake", sa.Float(), nullable=False),
        sa.Column("prob_real", sa.Float(), nullable=False),
        sa.Column("model_type", sa.String(length=64), nullable=False),
        sa.Column(
            "requires_human_review", sa.Boolean(), nullable=False, server_default=sa.text("0")
        ),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_index(op.f("ix_predictions_article_id"), "predictions", ["article_id"], unique=False)

    # 3. Feedbacks table
    op.create_table(
        "feedbacks",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column(
            "prediction_id",
            sa.String(length=36),
            sa.ForeignKey("predictions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("user_verdict", sa.String(length=32), nullable=False),
        sa.Column("comments", sa.Text(), nullable=True),
        sa.Column("is_reviewed", sa.Boolean(), nullable=False, server_default=sa.text("0")),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_index(
        op.f("ix_feedbacks_prediction_id"), "feedbacks", ["prediction_id"], unique=False
    )


def downgrade() -> None:
    op.drop_table("feedbacks")
    op.drop_table("predictions")
    op.drop_table("articles")
