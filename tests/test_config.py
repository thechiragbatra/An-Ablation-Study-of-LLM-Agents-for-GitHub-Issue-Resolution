from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from agent.config import ExperimentConfig, Settings
from tests.conftest import CONFIGS


def _base() -> dict[str, Any]:
    return ExperimentConfig.from_yaml(CONFIGS / "a0.yaml").model_dump()


@pytest.mark.parametrize("name", ["a0", "a1", "a2", "a3"])
def test_shipped_configs_load(name: str) -> None:
    cfg = ExperimentConfig.from_yaml(CONFIGS / f"{name}.yaml")
    assert cfg.variant == name.upper()
    assert not cfg.sandbox.network


def test_extra_keys_rejected() -> None:
    with pytest.raises(ValidationError, match="Extra inputs"):
        ExperimentConfig.model_validate({**_base(), "temperature": 0.2})


@pytest.mark.parametrize(
    ("overrides", "msg"),
    [
        ({"N": 2}, "N must be 1"),
        ({"variant": "A2", "N": 2}, "N must be 1"),
        ({"variant": "A3", "N": 2}, "N must be 1"),
        ({"variant": "A2", "D": 1}, "D must be 0"),
        ({"variant": "A1", "K": 3}, "K must be 1"),
        ({"variant": "A1", "D": 1}, "D must be 0"),
        ({"critic": {"enabled": True, "uses_sandbox": False}}, "no critic"),
        ({"critic": {"enabled": False, "uses_sandbox": True}}, "requires critic.enabled"),
        ({"budget": {"per_run_usd": 10, "per_experiment_usd": 1}}, "must not exceed"),
        ({"K": 6}, "less than or equal to 5"),
        ({"D": 3}, "less than or equal to 2"),
    ],
)
def test_variant_constraints(overrides: dict[str, Any], msg: str) -> None:
    with pytest.raises(ValidationError, match=msg):
        ExperimentConfig.model_validate({**_base(), **overrides})


def test_config_hash_stable_and_sensitive() -> None:
    a = ExperimentConfig.model_validate(_base())
    b = ExperimentConfig.model_validate(_base())
    c = ExperimentConfig.model_validate({**_base(), "seed": 1})
    assert a.config_hash() == b.config_hash() != c.config_hash()


@pytest.mark.parametrize(
    ("overrides", "rounds"),
    [
        ({}, 1),
        ({"variant": "A1", "N": 4}, 4),
        ({"variant": "A2", "K": 3, "critic": {"enabled": True}}, 1),
        ({"variant": "A3", "K": 3, "D": 2, "critic": {"enabled": True}}, 3),
        ({"variant": "A3", "D": 0}, 1),
    ],
)
def test_max_rounds(overrides: dict[str, Any], rounds: int) -> None:
    assert ExperimentConfig.model_validate({**_base(), **overrides}).max_rounds() == rounds


@pytest.mark.parametrize("variant", ["A2", "A3"])
def test_critic_can_be_disabled_for_ablation(variant: str) -> None:
    cfg = ExperimentConfig.model_validate({**_base(), "variant": variant, "K": 3})
    assert not cfg.critic.enabled


def test_recursion_limit_scales_with_caps() -> None:
    a0 = ExperimentConfig.model_validate(_base())
    a1 = ExperimentConfig.model_validate({**_base(), "variant": "A1", "N": 5})
    assert a1.recursion_limit() > a0.recursion_limit() > 4


def test_secret_never_rendered(monkeypatch: pytest.MonkeyPatch) -> None:
    secret = "sk-ant-test-0123456789abcdef"
    monkeypatch.setenv("ANTHROPIC_API_KEY", secret)
    s = Settings()
    assert s.anthropic_api_key is not None
    assert s.anthropic_api_key.get_secret_value() == secret
    assert secret not in repr(s)
    assert secret not in s.model_dump_json()


def test_from_yaml_rejects_non_mapping(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text("- just\n- a list\n")
    with pytest.raises(ValueError, match="mapping"):
        ExperimentConfig.from_yaml(p)
