"""Frozen data files: eval/data/subset_v1.json (benchmark) and eval/data/dev_set.json."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

DATA_DIR = Path(__file__).resolve().parent / "data"
DATASET = "SWE-bench/SWE-bench_Lite"
SPLIT = "test"
SetName = Literal["dev", "frozen"]
SET_FILES: dict[SetName, str] = {"dev": "dev_set.json", "frozen": "subset_v1.json"}


@dataclass(frozen=True)
class DataSet:
    name: SetName
    path: Path
    sha256: str
    meta: dict[str, Any]
    records: list[dict[str, Any]]

    @property
    def instance_ids(self) -> list[str]:
        return [r["instance_id"] for r in self.records]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump_data_file(path: Path, meta: dict[str, Any], records: list[Any]) -> None:
    """Deterministic serialization (sorted keys) so the checksum is meaningful."""
    payload = {"meta": meta, "instances": list(records)}
    path.write_text(json.dumps(payload, indent=1, sort_keys=True, ensure_ascii=False) + "\n")


def load_set(name: SetName, data_dir: Path = DATA_DIR) -> DataSet:
    path = data_dir / SET_FILES[name]
    payload = json.loads(path.read_text(encoding="utf-8"))
    return DataSet(name, path, file_sha256(path), payload["meta"], payload["instances"])
