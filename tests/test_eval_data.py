"""The benchmark subset is FROZEN; the dev set is disjoint from it."""

from __future__ import annotations

from collections import Counter

import pytest

from eval.data import DATASET, file_sha256, load_set
from eval.subset import allocate, draw, stratified_sample
from eval.swebench_adapter import GOLD_FIELDS, to_instance

# Changing eval/data/subset_v1.json invalidates the preregistration. If you really need a
# different subset, create subset_v2.json with a new preregistration instead.
SUBSET_V1_SHA256 = "2559e43747595cf9fcd13390b8d3220e04190fac11e1227abed7be2197d210bb"
DEV_SET_SHA256 = "fda951b1c4d4dd23b409799329a542bc6b367e93b6d756775b99025b6ed3ac59"
PINNED_REVISION = "b0dde1093fe417d83b7184254edf8199c1f0dff5"


def test_subset_is_frozen() -> None:
    frozen = load_set("frozen")
    assert file_sha256(frozen.path) == SUBSET_V1_SHA256, "FROZEN subset_v1.json was modified"
    assert frozen.meta["frozen"] is True


def test_dev_set_is_pinned() -> None:
    assert file_sha256(load_set("dev").path) == DEV_SET_SHA256


def test_dev_and_frozen_are_disjoint() -> None:
    frozen, dev = set(load_set("frozen").instance_ids), set(load_set("dev").instance_ids)
    assert not frozen & dev
    assert load_set("dev").meta["disjoint_from"] == "subset_v1.json"


def test_sizes_and_provenance() -> None:
    frozen, dev = load_set("frozen"), load_set("dev")
    assert 50 <= len(frozen.records) <= 100
    assert 10 <= len(dev.records) <= 15
    for ds in (frozen, dev):
        assert ds.meta["dataset"] == DATASET
        assert ds.meta["revision"] == PINNED_REVISION
        assert isinstance(ds.meta["seed"], int)
        assert len(set(ds.instance_ids)) == len(ds.records)


def test_records_become_agent_safe_instances() -> None:
    for rec in load_set("frozen").records + load_set("dev").records:
        inst = to_instance(rec)
        dumped = inst.model_dump_json()
        assert rec["patch"] not in dumped
        assert rec["test_patch"] not in dumped
        assert inst.image == rec["image"]
        assert inst.workdir == "/testbed"
        assert not set(GOLD_FIELDS) & set(inst.model_dump())


# --------------------------------------------------------------------------- sampling logic


def _synthetic(n_by_repo: dict[str, int]) -> list[dict[str, str]]:
    return [
        {"instance_id": f"{repo}-{i:03d}", "repo": repo}
        for repo, n in n_by_repo.items()
        for i in range(n)
    ]


def test_allocation_is_proportional_and_exact() -> None:
    quota = allocate({"django": 114, "sympy": 77, "other": 109}, 75)
    assert sum(quota.values()) == 75
    assert quota == {"django": 29, "sympy": 19, "other": 27}
    with pytest.raises(ValueError, match="size"):
        allocate({"a": 3}, 4)


def test_sampling_is_deterministic_stratified_and_disjoint() -> None:
    records = _synthetic({"django": 114, "sympy": 77, "flask": 3, "requests": 6})
    a = draw(records, 75, 15, seed=7)
    b = draw(list(reversed(records)), 75, 15, seed=7)  # input order must not matter
    assert [r["instance_id"] for r in a[0]] == [r["instance_id"] for r in b[0]]
    subset, dev = a
    assert len(subset) == 75
    assert len(dev) == 15
    assert not {r["instance_id"] for r in subset} & {r["instance_id"] for r in dev}
    counts = Counter(r["repo"] for r in subset)
    assert counts["django"] == 43
    assert counts["sympy"] == 29
    other = stratified_sample(records, 75, seed=8)
    assert [r["instance_id"] for r in other] != [r["instance_id"] for r in subset]
