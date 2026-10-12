"""Dataset loader and synthetic stub generation for tests."""

from pathlib import Path
from typing import Any

import pandas as pd


def create_synthetic_news_dataset(num_samples: int = 20) -> pd.DataFrame:
    """Generate a minimal synthetic dataset strictly for unit testing and CI verification.

    NOTE: This is NOT a real-world benchmark dataset. It provides structured text stubs
    with known dummy patterns so unit tests and model pipelines can execute deterministically.

    Args:
        num_samples: Total number of synthetic examples to produce.

    Returns:
        pd.DataFrame containing columns ['id', 'title', 'text', 'label'].
    """
    fake_stubs = [
        (
            "Shocking Miracle Cure Discovered Overnight By Secret Scientists!",
            "You will not believe this secret conspiracy. Doctors hate this one weird trick "
            "that completely reverses all human diseases in twenty-four hours. Share before deleted!",
        ),
        (
            "Alien Spaceship Lands in Central Park, Government Confirms Coverup",
            "Anonymous whistleblowers claim an extraterrestrial vessel touched down yesterday. "
            "Authorities are allegedly silencing journalists and scrubbing satellite imagery.",
        ),
        (
            "Celebrity Billionaire Secretly Buys Moon and Declares Independence",
            "Exclusive leaked documents reveal the billionaire has established sovereign control "
            "over lunar craters without international treaty approval.",
        ),
    ]

    real_stubs = [
        (
            "Federal Reserve Announces Modest Interest Rate Adjustment",
            "The Federal Open Market Committee concluded its two-day policy meeting today, "
            "announcing a benchmark interest rate adjustment of twenty-five basis points to curb inflation.",
        ),
        (
            "Astronomers Publish New Data from James Webb Space Telescope",
            "In a peer-reviewed study published in Nature Astronomy, researchers documented "
            "spectroscopic observations of distant galaxy formations dating back to the early universe.",
        ),
        (
            "Municipal Council Approves Infrastructure Budget for Public Transit",
            "The city council voted unanimously to allocate capital funding for the expansion "
            "of regional bus routes and railway track maintenance over the next fiscal year.",
        ),
    ]

    records: list[dict[str, Any]] = []
    for i in range(num_samples):
        is_fake = (i % 2) == 1
        source_stubs = fake_stubs if is_fake else real_stubs
        title, text = source_stubs[(i // 2) % len(source_stubs)]
        records.append(
            {
                "id": f"synthetic-{i + 1:04d}",
                "title": f"{title} (Sample {i + 1})",
                "text": f"{text} Reference ID: SYNTH-{i + 1}.",
                "label": 1 if is_fake else 0,  # 1 = Fake, 0 = Real
            }
        )

    return pd.DataFrame(records)


class DataLoader:
    """Loads and validates news datasets from local or remote file paths."""

    EXPECTED_COLUMNS = {"title", "text", "label"}

    @classmethod
    def load_from_csv(cls, file_path: str | Path) -> pd.DataFrame:
        """Load tabular dataset and validate required schema.

        TODO: Connect to enterprise data lakes or download public benchmarks
        (e.g., WELFake, ISOT, or LIAR datasets) via DVC remote storage.

        Args:
            file_path: Path to CSV dataset.

        Returns:
            Validated pandas DataFrame.

        Raises:
            FileNotFoundError: If the specified file does not exist.
            ValueError: If expected columns are missing.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found at: {path}")

        df = pd.read_csv(path)
        missing_cols = cls.EXPECTED_COLUMNS - set(df.columns)
        if missing_cols:
            raise ValueError(
                f"Dataset at {path} is missing required columns: {missing_cols}. "
                f"Expected: {cls.EXPECTED_COLUMNS}"
            )

        # Drop null values in core text fields
        df = df.dropna(subset=["text"]).reset_index(drop=True)
        return df
