"""Small, generic helpers shared across modules.

Nothing brand-specific or LLM-specific belongs here -- just I/O and parsing
utilities that multiple modules would otherwise duplicate.
"""
from __future__ import annotations

import csv
import json
import random
from collections.abc import Iterable, Iterator
from datetime import datetime
from pathlib import Path
from typing import TypeVar

import numpy as np
import yaml
from pydantic import BaseModel

from .schemas import GoldenExample, IntentDefinition

# GoldenExample columns that are Optional/Literal-typed: csv.DictWriter
# serializes Python `None` as an empty string, but csv.DictReader reads it
# back as `""`, which pydantic rejects for these fields (neither an Enum
# nor a Literal accepts an empty string as None). Restoring `None` here
# before validation makes the CSV round trip lossless. This bit us twice
# from two separate ad hoc CSV readers before being pulled out to one place.
_GOLDEN_CSV_NULLABLE_FIELDS = ("llm_suggested_intent", "llm_suggested_action", "split")

# Twitter's raw export format, e.g. "Tue Oct 31 22:10:47 +0000 2017".
_TWITTER_DATETIME_FORMAT = "%a %b %d %H:%M:%S %z %Y"

M = TypeVar("M", bound=BaseModel)


def parse_twitter_datetime(raw: str) -> datetime:
    """Parse TWCS's `created_at` format into a timezone-aware datetime.

    Raises ValueError on malformed input rather than guessing -- callers
    that need to tolerate bad timestamps should catch it explicitly.
    """
    return datetime.strptime(raw, _TWITTER_DATETIME_FORMAT)


def set_global_seed(seed: int) -> None:
    """Seed every RNG this codebase touches, for reproducible sampling/splits."""
    random.seed(seed)
    np.random.seed(seed)


def write_jsonl(path: Path, records: Iterable[BaseModel]) -> int:
    """Write pydantic models to `path` as JSON Lines. Returns the row count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(record.model_dump_json())
            f.write("\n")
            count += 1
    return count


def read_jsonl(path: Path, model: type[M]) -> Iterator[M]:
    """Stream-parse a JSON Lines file into instances of `model`."""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield model.model_validate(json.loads(line))


def load_intents(path: Path) -> list[IntentDefinition]:
    """Load and validate the intent taxonomy from intents.yaml."""
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return [IntentDefinition.model_validate(item) for item in raw]


def read_golden_csv(path: Path) -> list[GoldenExample]:
    """Read golden_set.csv into validated GoldenExample rows.

    The one place this happens -- `label_tool.py` and `baselines.py` both
    call this rather than each parsing the CSV themselves, after the same
    empty-string-vs-None bug (see `_GOLDEN_CSV_NULLABLE_FIELDS`) was fixed
    twice in two separate ad hoc readers.
    """
    with path.open(newline="", encoding="utf-8") as f:
        rows = []
        for row in csv.DictReader(f):
            for field in _GOLDEN_CSV_NULLABLE_FIELDS:
                if row.get(field) == "":
                    row[field] = None
            rows.append(GoldenExample.model_validate(row))
        return rows
