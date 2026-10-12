"""Pydantic schemas for article input and processing."""

from pydantic import BaseModel, Field, model_validator


class ArticleInput(BaseModel):
    """Article payload submitted for fake news analysis."""

    text: str | None = Field(
        default=None,
        description="Raw article body or text content.",
        min_length=10,
    )
    url: str | None = Field(
        default=None,
        description="Direct web URL to scrape the article from.",
    )
    title: str | None = Field(
        default=None,
        description="Optional article headline.",
    )
    model_type: str | None = Field(
        default=None,
        description="Optional model selector: 'baseline' or 'transformer'.",
    )

    @model_validator(mode="after")
    def check_at_least_one_source(self) -> "ArticleInput":
        """Validate that either raw text or a valid URL is provided."""
        if not self.text and not self.url:
            raise ValueError("Either 'text' or 'url' must be provided.")
        return self


class ArticleResponse(BaseModel):
    """Article representation stored in database."""

    id: str
    url: str | None = None
    title: str | None = None
    raw_text: str
    cleaned_text: str

    model_config = {"from_attributes": True}
