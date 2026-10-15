"""Configuration: secrets from environment, experiment knobs from YAML.

The graph variant is selected here and only here; code never branches on anything else.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Variant = Literal["A0", "A1", "A2", "A3"]
Effort = Literal["low", "medium", "high", "xhigh", "max"]

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PRICING_PATH = REPO_ROOT / "configs" / "pricing.yaml"


class Settings(BaseSettings):
    """Secrets and machine-local settings. Read from the environment only."""

    model_config = SettingsConfigDict(env_file=None, extra="ignore")

    anthropic_api_key: SecretStr | None = None
    docker_host: str | None = None


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CriticConfig(_Strict):
    enabled: bool = False
    uses_sandbox: bool = False


class BudgetConfig(_Strict):
    per_run_usd: float = Field(gt=0)
    per_experiment_usd: float = Field(gt=0)

    @model_validator(mode="after")
    def _run_within_experiment(self) -> Self:
        if self.per_run_usd > self.per_experiment_usd:
            raise ValueError("per_run_usd must not exceed per_experiment_usd")
        return self


class PublisherConfig(_Strict):
    """Live pilot only. When enabled, a sandbox-verified run is written as a LOCAL pending
    PR package; nothing in the agent ever contacts GitHub (see pilot/publish.py)."""

    enabled: bool = False
    pending_dir: Path = Path("out/pending")


class SandboxConfig(_Strict):
    backend: Literal["docker", "e2b"] = "docker"
    image: str
    timeout_s: int = Field(default=300, gt=0)
    mem_limit: str = "2g"
    cpus: float = Field(default=2.0, gt=0)
    network: bool = False


class ExperimentConfig(_Strict):
    name: str
    variant: Variant
    model: str = "claude-opus-5-5"
    effort: Effort = "high"
    max_tokens: int = Field(default=16000, gt=0)
    K: int = Field(default=1, ge=1, le=5)
    D: int = Field(default=0, ge=0, le=2)
    N: int = Field(default=1, ge=1, le=10)
    seed: int = 0
    max_localized_files: int = Field(default=5, ge=1)
    critic: CriticConfig = CriticConfig()
    budget: BudgetConfig
    sandbox: SandboxConfig
    trace_dir: Path = Path("out/traces")
    publisher: PublisherConfig = PublisherConfig()

    @model_validator(mode="after")
    def _variant_constraints(self) -> Self:
        v = self.variant
        if self.critic.uses_sandbox and not self.critic.enabled:
            raise ValueError("critic.uses_sandbox requires critic.enabled")
        if v in ("A0", "A1"):
            if self.K != 1:
                raise ValueError(f"{v} generates a single candidate; K must be 1")
            if self.critic.enabled:
                raise ValueError(f"{v} has no critic; critic.enabled must be false")
        if v in ("A0", "A1", "A2") and self.D != 0:
            raise ValueError(f"{v} has no reflection loop; D must be 0")
        if v in ("A0", "A2", "A3") and self.N != 1:
            raise ValueError(f"{v} does not use plain retries; N must be 1")
        return self

    def max_rounds(self) -> int:
        """Upper bound on architect rounds; every graph cycle is guarded by it."""
        if self.variant == "A1":
            return self.N
        if self.variant == "A3":
            return self.D + 1
        return 1

    @classmethod
    def from_yaml(cls, path: Path) -> ExperimentConfig:
        data = yaml.safe_load(path.read_text())
        if not isinstance(data, dict):
            raise ValueError(f"{path}: expected a mapping at top level")
        return cls.model_validate(data)

    def config_hash(self) -> str:
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()

    def recursion_limit(self) -> int:
        """Backstop for LangGraph; the real caps are max_rounds() and the budget guard.

        Per round at most: architect, patcher, select, validator, critic_reflect."""
        return 5 * self.max_rounds() + 10
