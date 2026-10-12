"""Data ingestion, preprocessing, and web scraping utilities."""

from fake_news_detector.data.loader import DataLoader, create_synthetic_news_dataset
from fake_news_detector.data.preprocessor import TextPreprocessor
from fake_news_detector.data.scraper import ArticleScraper

__all__ = [
    "DataLoader",
    "create_synthetic_news_dataset",
    "TextPreprocessor",
    "ArticleScraper",
]
