"""
Agent Registry — discover, register, and manage agents across all clouds.

The registry is the source of truth for every AI agent in the enterprise:
which cloud it runs on, what industry it serves, what SLA it commits to,
and what compliance tags it carries. Orchestrate consults this registry
for every routing decision.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from src.utils.config import get_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


class AgentStatus(str, Enum):
    ACTIVE = "active"
    DEGRADED = "degraded"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"


@dataclass
class AgentHealth:
    agent_id: str
    status: AgentStatus
    success_rate_24h: float = 1.0
    avg_latency_ms: float = 0.0
    error_count_24h: int = 0
    last_checked: float = field(default_factory=time.time)
    clouds_healthy: list[str] = field(default_factory=list)


@dataclass
class RegisteredAgent:
    """A fully described agent entry from the registry."""

    id: str
    name: str
    industry: str
    description: str
    clouds: list[str]
    orchestration: str
    graph_module: str
    trigger_intent: list[str]
    sla_seconds: int
    requires_human_approval: bool = False
    compliance_tags: list[str] = field(default_factory=list)
    status: AgentStatus = AgentStatus.ACTIVE
    health: AgentHealth | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RegisteredAgent":
        return cls(
            id=data["id"],
            name=data["name"],
            industry=data.get("industry", ""),
            description=data.get("description", ""),
            clouds=data.get("clouds", ["aws"]),
            orchestration=data.get("orchestration", "sequential"),
            graph_module=data.get("graph_module", ""),
            trigger_intent=data.get("trigger_intent", []),
            sla_seconds=data.get("sla_seconds", 60),
            requires_human_approval=data.get("requires_human_approval", False),
            compliance_tags=data.get("compliance_tags", []),
        )

    def to_skill_card(self) -> dict[str, Any]:
        """Return the Orchestrate skill card representation."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "industry": self.industry,
            "clouds": self.clouds,
            "sla_seconds": self.sla_seconds,
            "requires_approval": self.requires_human_approval,
            "compliance": self.compliance_tags,
            "status": self.status.value,
            "trigger_phrases": self.trigger_intent[:3],
        }


class AgentRegistry:
    """
    In-process agent registry backed by orchestrate_config.yaml.

    Provides lookup, filtering, health tracking, and skill card
    generation for all registered agents across all clouds.

    Example:
        registry = AgentRegistry()

        # List all Financial Services agents
        agents = registry.list_by_industry("financial_services")
        for agent in agents:
            print(agent.name, "→", agent.clouds)

        # Find agents that run on GCP
        gcp_agents = registry.list_by_cloud("gcp")

        # Get a specific agent
        agent = registry.get("fs_loan_origination")
        print(agent.compliance_tags)
    """

    def __init__(self) -> None:
        cfg = get_config()
        self._agents: dict[str, RegisteredAgent] = {
            entry["id"]: RegisteredAgent.from_dict(entry)
            for entry in cfg.agent_registry
        }
        self._health: dict[str, AgentHealth] = {}

        logger.info("AgentRegistry loaded", count=len(self._agents))

    def get(self, agent_id: str) -> RegisteredAgent:
        if agent_id not in self._agents:
            raise KeyError(f"Agent not found: {agent_id}")
        return self._agents[agent_id]

    def list_all(self) -> list[RegisteredAgent]:
        return list(self._agents.values())

    def list_by_industry(self, industry: str) -> list[RegisteredAgent]:
        return [a for a in self._agents.values() if a.industry == industry]

    def list_by_cloud(self, cloud: str) -> list[RegisteredAgent]:
        return [a for a in self._agents.values() if cloud in a.clouds]

    def list_by_compliance_tag(self, tag: str) -> list[RegisteredAgent]:
        return [a for a in self._agents.values() if tag in a.compliance_tags]

    def list_requiring_human_approval(self) -> list[RegisteredAgent]:
        return [a for a in self._agents.values() if a.requires_human_approval]

    def industries(self) -> list[str]:
        return list(dict.fromkeys(a.industry for a in self._agents.values()))

    def clouds(self) -> list[str]:
        all_clouds: list[str] = []
        for agent in self._agents.values():
            all_clouds.extend(agent.clouds)
        return list(dict.fromkeys(all_clouds))

    def skill_catalog(self) -> list[dict[str, Any]]:
        """Return full Orchestrate skill catalog — all registered agents."""
        return [agent.to_skill_card() for agent in self._agents.values()]

    def update_health(self, agent_id: str, health: AgentHealth) -> None:
        """Update the health record for an agent."""
        if agent_id in self._agents:
            self._agents[agent_id].health = health
            self._agents[agent_id].status = health.status
            self._health[agent_id] = health
            logger.debug("Agent health updated", agent_id=agent_id, status=health.status.value)

    def summary(self) -> dict[str, Any]:
        """Return a summary of the registry state."""
        all_agents = list(self._agents.values())
        return {
            "total_agents": len(all_agents),
            "by_industry": {
                ind: len(self.list_by_industry(ind)) for ind in self.industries()
            },
            "by_cloud": {
                cloud: len(self.list_by_cloud(cloud)) for cloud in self.clouds()
            },
            "requiring_human_approval": len(self.list_requiring_human_approval()),
            "active": sum(1 for a in all_agents if a.status == AgentStatus.ACTIVE),
            "degraded": sum(1 for a in all_agents if a.status == AgentStatus.DEGRADED),
            "offline": sum(1 for a in all_agents if a.status == AgentStatus.OFFLINE),
        }
