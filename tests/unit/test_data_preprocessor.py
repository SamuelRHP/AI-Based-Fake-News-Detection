"""Unit tests for text preprocessing, cleaning, and sentence segmentation."""

from fake_news_detector.data.preprocessor import TextPreprocessor


def test_clean_text_removes_html_and_urls():
    preprocessor = TextPreprocessor(lowercase=True, strip_html=True)
    raw = "<p>Shocking update at https://example.com/breaking! Visit www.fake.com now.</p>"
    cleaned = preprocessor.clean_text(raw)
    assert "<p>" not in cleaned
    assert "https://" not in cleaned
    assert "www.fake.com" not in cleaned
    assert "shocking update at" in cleaned


def test_clean_text_normalizes_whitespace():
    preprocessor = TextPreprocessor()
    raw = "Headline   with   \n\n  erratic    spacing.  "
    cleaned = preprocessor.clean_text(raw)
    assert cleaned == "headline with erratic spacing."


def test_extract_sentences():
    preprocessor = TextPreprocessor()
    article = "First sentence here. Second breaking claim! Is this third sentence? Yes."
    sentences = preprocessor.extract_sentences(article)
    assert len(sentences) == 4
    assert sentences[0] == "First sentence here."
    assert sentences[1] == "Second breaking claim!"


def test_clean_text_handles_empty_or_none():
    preprocessor = TextPreprocessor()
    assert preprocessor.clean_text("") == ""
    assert preprocessor.clean_text(None) == ""  # type: ignore[arg-type]
