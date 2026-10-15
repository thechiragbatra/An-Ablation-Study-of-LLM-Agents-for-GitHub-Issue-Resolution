"""Draw the FROZEN benchmark subset and the disjoint dev set from SWE-bench Lite.

    uv run python -m eval.subset --size 75 --dev-size 15 --seed 20261015

Sampling is stratified by repository with proportional (largest-remainder) allocation,
seeded, over instance ids sorted for determinism. The dev set is drawn from the
instances NOT in the subset. Output files hold full records plus provenance, so they
are self-contained. Existing files are never overwritten: freezing is one-way, and a
changed subset needs a new version name (subset_v2.json) and a new preregistration.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from eval.data import DATA_DIR, DATASET, SPLIT, dump_data_file

Record = Mapping[str, Any]


def allocate(counts: Mapping[str, int], size: int) -> dict[str, int]:
    """Proportional allocation with largest remainders (ties broken by repo name)."""
    total = sum(counts.values())
    if not 0 < size <= total:
        raise ValueError(f"size must be in 1..{total}, got {size}")
    exact = {k: size * n / total for k, n in counts.items()}
    quota = {k: int(v) for k, v in exact.items()}
    leftover = size - sum(quota.values())
    for k in sorted(exact, key=lambda k: (-(exact[k] - quota[k]), k))[:leftover]:
        quota[k] += 1
    return quota


def stratified_sample(records: Sequence[Record], size: int, seed: int) -> list[Record]:
    by_repo: dict[str, list[Record]] = defaultdict(list)
    for r in sorted(records, key=lambda r: r["instance_id"]):
        by_repo[r["repo"]].append(r)
    quota = allocate({k: len(v) for k, v in by_repo.items()}, size)
    rng = random.Random(seed)
    chosen: list[Record] = []
    for repo in sorted(by_repo):
        chosen.extend(rng.sample(by_repo[repo], quota[repo]))
    return sorted(chosen, key=lambda r: r["instance_id"])


def draw(
    records: Sequence[Record], size: int, dev_size: int, seed: int
) -> tuple[list[Record], list[Record]]:
    subset = stratified_sample(records, size, seed)
    taken = {r["instance_id"] for r in subset}
    rest = [r for r in records if r["instance_id"] not in taken]
    dev = stratified_sample(rest, dev_size, seed + 1)
    return subset, dev


def _load_lite(revision: str | None) -> tuple[list[dict[str, Any]], str]:
    # Lazy: only needed when (re)drawing data; `datasets` is in the eval group.
    from datasets import load_dataset  # noqa: PLC0415
    from huggingface_hub import HfApi  # noqa: PLC0415

    rev = revision or HfApi().dataset_info(DATASET).sha
    if rev is None:
        raise RuntimeError("could not resolve dataset revision")
    ds = load_dataset(DATASET, split=SPLIT, revision=rev)
    return [dict(r) for r in ds], rev


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    ap.add_argument("--size", type=int, default=75)
    ap.add_argument("--dev-size", type=int, default=15)
    ap.add_argument("--seed", type=int, default=20261015)
    ap.add_argument("--revision", help="HF dataset revision (default: resolve current)")
    ap.add_argument("--version", default="v1")
    ap.add_argument("--out-dir", type=Path, default=DATA_DIR)
    args = ap.parse_args(argv)
    if not 50 <= args.size <= 100:
        ap.error("the preregistered design uses 50-100 benchmark instances")
    if not 10 <= args.dev_size <= 15:
        ap.error("the dev set must have 10-15 instances")

    subset_path = args.out_dir / f"subset_{args.version}.json"
    dev_path = args.out_dir / "dev_set.json"
    for p in (subset_path, dev_path):
        if p.exists():
            print(f"refusing to overwrite frozen file {p}", file=sys.stderr)
            return 1

    records, revision = _load_lite(args.revision)
    subset, dev = draw(records, args.size, args.dev_size, args.seed)
    common = {
        "dataset": DATASET,
        "split": SPLIT,
        "revision": revision,
        "seed": args.seed,
        "sampling": "stratified by repo, proportional largest-remainder allocation",
        "created": datetime.now(UTC).date().isoformat(),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dump_data_file(
        subset_path,
        {**common, "name": f"subset_{args.version}", "frozen": True, "purpose": "benchmark"},
        subset,
    )
    dump_data_file(
        dev_path,
        {**common, "name": "dev_set", "frozen": True, "purpose": "tuning only",
         "seed": args.seed + 1, "disjoint_from": subset_path.name},
        dev,
    )  # fmt: skip
    for name, recs in (("subset", subset), ("dev", dev)):
        repos: dict[str, int] = defaultdict(int)
        for r in recs:
            repos[r["repo"]] += 1
        print(f"{name}: {len(recs)} instances  {json.dumps(dict(sorted(repos.items())))}")
    print(f"revision {revision}\nwrote {subset_path} and {dev_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
