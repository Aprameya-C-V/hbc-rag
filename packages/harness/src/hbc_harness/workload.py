from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field


class DatasetQuestion(BaseModel):
    qid: str
    dataset: str
    question: str
    answer: str
    support_keys: list[str] = Field(default_factory=list)


def load_questions(path: str | Path) -> list[DatasetQuestion]:
    rows = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(DatasetQuestion(**json.loads(line)))
    return rows
