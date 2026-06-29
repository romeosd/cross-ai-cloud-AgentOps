"""
Configuration loader for Cross-Cloud AgentOps.
Loads orchestrate_config.yaml with environment variable substitution.
"""
from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


def _substitute_env_vars(value: str) -> str:
    pattern = re.compile(r"\$\{([^}:]+)(?::-(.*?))?\}")
    def replacer(match: re.Match) -> str:
        var_name = match.group(1)
        default = match.group(2) if match.group(2) is not None else ""
        return os.environ.get(var_name, default)
    return pattern.sub(replacer, value)


def _process(data: Any) -> Any:
    if isinstance(data, dict):
        return {k: _process(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_process(i) for i in data]
    elif isinstance(data, str):
        return _substitute_env_vars(data)
    return data


@lru_cache(maxsize=1)
def load_config(path: str | None = None) -> dict[str, Any]:
    if path is None:
        root = Path(__file__).parent.parent.parent
        path = str(root / "config" / "orchestrate_config.yaml")
    with open(path) as f:
        return _process(yaml.safe_load(f))


class OrchestrateConfig(BaseModel):
    api_url: str = Field(default="")
    instance_id: str = Field(default="")
    api_key: str = Field(default="")
    environment: str = Field(default="production")
    governance: dict[str, Any] = Field(default_factory=dict)
    routing: dict[str, Any] = Field(default_factory=dict)


class CloudConfig(BaseModel):
    aws: dict[str, Any] = Field(default_factory=dict)
    azure: dict[str, Any] = Field(default_factory=dict)
    gcp: dict[str, Any] = Field(default_factory=dict)


class AgentOpsConfig(BaseModel):
    orchestrate: OrchestrateConfig = Field(default_factory=OrchestrateConfig)
    clouds: CloudConfig = Field(default_factory=CloudConfig)
    agent_registry: list[dict[str, Any]] = Field(default_factory=list)
    agentops: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_yaml(cls) -> "AgentOpsConfig":
        raw = load_config()
        return cls(
            orchestrate=OrchestrateConfig(**raw.get("orchestrate", {})),
            clouds=CloudConfig(**raw.get("clouds", {})),
            agent_registry=raw.get("agent_registry", []),
            agentops=raw.get("agentops", {}),
        )

    def get_agent(self, agent_id: str) -> dict[str, Any]:
        for agent in self.agent_registry:
            if agent["id"] == agent_id:
                return agent
        raise KeyError(f"Agent '{agent_id}' not found in registry.")

    def agents_by_industry(self, industry: str) -> list[dict[str, Any]]:
        return [a for a in self.agent_registry if a.get("industry") == industry]


_cfg: AgentOpsConfig | None = None

def get_config() -> AgentOpsConfig:
    global _cfg
    if _cfg is None:
        _cfg = AgentOpsConfig.from_yaml()
    return _cfg
