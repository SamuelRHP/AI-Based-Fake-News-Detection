"""Web scraping and article extraction from external URLs."""

import logging
from typing import TypedDict

import requests
from bs4 import BeautifulSoup

from fake_news_detector.config import settings

logger = logging.getLogger(__name__)


class ScrapedArticle(TypedDict):
    """Structured representation of extracted article content."""

    url: str
    title: str
    text: str
    success: bool
    error: str | None


class ArticleScraper:
    """Extracts headline and main article body from target URLs."""

    def __init__(self, timeout: int | None = None, user_agent: str | None = None) -> None:
        """Initialize scraper with network configurations."""
        self.timeout = timeout or settings.REQUEST_TIMEOUT_SECONDS
        self.user_agent = user_agent or settings.USER_AGENT
        self.headers = {"User-Agent": self.user_agent}

    def fetch_and_extract(self, url: str) -> ScrapedArticle:
        """Fetch HTML content from a URL and extract title and body text.

        Args:
            url: The HTTP/HTTPS web address of the news article.

        Returns:
            ScrapedArticle dictionary with extracted text, title, and status.
        """
        if not url or not (url.startswith("http://") or url.startswith("https://")):
            return {
                "url": url,
                "title": "",
                "text": "",
                "success": False,
                "error": "Invalid URL protocol. URL must start with http:// or https://",
            }

        try:
            response = requests.get(url, headers=self.headers, timeout=self.timeout)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            # Remove scripts, styles, navigations, and footer noise
            for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
                element.extract()

            # Extract title
            title_node = soup.find("title") or soup.find("h1")
            title = title_node.get_text(strip=True) if title_node else ""

            # Extract main article body by analyzing paragraph tags
            article_node = soup.find("article") or soup.find("main") or soup.body
            paragraphs = []
            if article_node:
                for p in article_node.find_all("p"):
                    text_chunk = p.get_text(strip=True)
                    # Filter short boilerplate lines
                    if len(text_chunk.split()) >= 5:
                        paragraphs.append(text_chunk)

            article_text = " ".join(paragraphs)

            if not article_text:
                return {
                    "url": url,
                    "title": title,
                    "text": "",
                    "success": False,
                    "error": "No meaningful article body found in the target page.",
                }

            return {
                "url": url,
                "title": title,
                "text": article_text,
                "success": True,
                "error": None,
            }

        except requests.RequestException as exc:
            logger.warning(f"Failed to scrape URL {url}: {exc}")
            return {
                "url": url,
                "title": "",
                "text": "",
                "success": False,
                "error": f"Network error during article fetch: {str(exc)}",
            }
