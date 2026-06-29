"""
Cost Tracker — per-agent, per-cloud, per-industry cost attribution.

Tracks token spend and USD costs across all three clouds and surfaces
chargeback reports, budget alerts, and cost optimisation recommendations.
"""
from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from src.utils.config import get_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CostEntry:
    timestamp: float
    agent_id: str
    cloud: str
    industry: str
    tokens_input: int
    tokens_output: int
    cost_usd: float
    workflow_id: str = ""
    business_unit: str = ""


@dataclass
class CostReport:
    period_label: str
    total_usd: float
    by_cloud: dict[str, float] = field(default_factory=dict)
    by_industry: dict[str, float] = field(default_factory=dict)
    by_agent: dict[str, float] = field(default_factory=dict)
    by_business_unit: dict[str, float] = field(default_factory=dict)
    token_breakdown: dict[str, int] = field(default_factory=dict)
    top_5_agents: list[dict[str, Any]] = field(default_factory=list)
    cost_optimisation_tips: list[str] = field(default_factory=list)
    projected_monthly_usd: float = 0.0


class CostTracker:
    """
    Enterprise cost attribution for multi-cloud AI agent operations.

    Tracks every token and dollar spent across AWS Bedrock, Azure OpenAI,
    and Vertex AI, with chargeback reporting by business unit, industry,
    and individual agent.

    Example:
        tracker = CostTracker()
        tracker.record(agent_id="fs_loan_origination", cloud="aws",
                       industry="financial_services", tokens_input=800,
                       tokens_output=600, cost_usd=0.042, business_unit="Retail_Banking")

        report = tracker.monthly_report()
        print(f"Total: ${report.total_usd:.2f}")
        print("By cloud:", report.by_cloud)
    """

    # Cost per 1K tokens by cloud (USD, ap-southeast-2 / australiaeast / australia-southeast1)
    CLOUD_RATES: dict[str, dict[str, float]] = {
        "aws":   {"input": 0.003,    "output": 0.015},    # Claude Sonnet
        "azure": {"input": 0.0025,   "output": 0.010},    # GPT-4o
        "gcp":   {"input": 0.000075, "output": 0.000300}, # Gemini 2.0 Flash per 1K
    }

    def __init__(self) -> None:
        self._entries: list[CostEntry] = []
        self._cfg = get_config()
        self._daily_threshold = self._cfg.agentops.get("cost_alert_threshold_usd", 100.0)
        logger.info("CostTracker initialised", daily_threshold=self._daily_threshold)

    def record(
        self,
        agent_id: str,
        cloud: str,
        industry: str,
        tokens_input: int,
        tokens_output: int,
        cost_usd: float | None = None,
        workflow_id: str = "",
        business_unit: str = "",
    ) -> CostEntry:
        """Record a cost entry, auto-computing cost if not provided."""
        if cost_usd is None:
            rates = self.CLOUD_RATES.get(cloud, {"input": 0.003, "output": 0.015})
            cost_usd = (tokens_input / 1000 * rates["input"]) + (tokens_output / 1000 * rates["output"])

        entry = CostEntry(
            timestamp=time.time(),
            agent_id=agent_id,
            cloud=cloud,
            industry=industry,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            cost_usd=cost_usd,
            workflow_id=workflow_id,
            business_unit=business_unit,
        )
        self._entries.append(entry)
        return entry

    def daily_report(self, date_offset_days: int = 0) -> CostReport:
        """Generate a cost report for a specific day (0 = today)."""
        now = time.time()
        start = now - (date_offset_days + 1) * 86400
        end = now - date_offset_days * 86400
        entries = [e for e in self._entries if start <= e.timestamp < end]
        return self._build_report(entries, f"Daily (T-{date_offset_days})")

    def monthly_report(self) -> CostReport:
        """Generate a cost report for the current month."""
        cutoff = time.time() - 30 * 86400
        entries = [e for e in self._entries if e.timestamp >= cutoff]
        return self._build_report(entries, "Last 30 days")

    def chargeback_report(self, business_unit: str) -> CostReport:
        """Cost report filtered to a specific business unit."""
        entries = [e for e in self._entries if e.business_unit == business_unit]
        return self._build_report(entries, f"Chargeback: {business_unit}")

    def estimate_cost(self, cloud: str, tokens_input: int, tokens_output: int) -> float:
        """Estimate cost before invoking an agent."""
        rates = self.CLOUD_RATES.get(cloud, {"input": 0.003, "output": 0.015})
        return (tokens_input / 1000 * rates["input"]) + (tokens_output / 1000 * rates["output"])

    def cheapest_cloud_for_tokens(self, tokens_input: int, tokens_output: int) -> str:
        """Return the cloud with lowest estimated cost for a given token count."""
        costs = {
            cloud: self.estimate_cost(cloud, tokens_input, tokens_output)
            for cloud in self.CLOUD_RATES
        }
        return min(costs, key=lambda c: costs[c])

    def _build_report(self, entries: list[CostEntry], label: str) -> CostReport:
        if not entries:
            return CostReport(period_label=label, total_usd=0.0)

        total = sum(e.cost_usd for e in entries)
        by_cloud: dict[str, float] = defaultdict(float)
        by_industry: dict[str, float] = defaultdict(float)
        by_agent: dict[str, float] = defaultdict(float)
        by_bu: dict[str, float] = defaultdict(float)
        tokens_in = sum(e.tokens_input for e in entries)
        tokens_out = sum(e.tokens_output for e in entries)

        for e in entries:
            by_cloud[e.cloud] += e.cost_usd
            by_industry[e.industry] += e.cost_usd
            by_agent[e.agent_id] += e.cost_usd
            if e.business_unit:
                by_bu[e.business_unit] += e.cost_usd

        top_5 = sorted(
            [{"agent_id": k, "cost_usd": round(v, 4)} for k, v in by_agent.items()],
            key=lambda x: x["cost_usd"],
            reverse=True,
        )[:5]

        # Optimisation tips
        tips: list[str] = []
        cloud_shares = {k: v / total for k, v in by_cloud.items()}
        if cloud_shares.get("aws", 0) > 0.6:
            tips.append("AWS is >60% of cost — consider routing simpler tasks to GCP Gemini Flash (50x cheaper per token)")
        if cloud_shares.get("gcp", 0) < 0.1 and "gcp" in by_cloud:
            tips.append("GCP underutilised — Gemini Flash is optimal for high-volume classification and extraction tasks")
        if tokens_in > tokens_out * 5:
            tips.append("High input:output ratio — consider prompt compression or caching for repeated context")

        # Project to monthly
        duration_days = (max(e.timestamp for e in entries) - min(e.timestamp for e in entries)) / 86400
        projected = total / max(duration_days, 1) * 30 if duration_days > 0 else total * 30

        return CostReport(
            period_label=label,
            total_usd=round(total, 4),
            by_cloud={k: round(v, 4) for k, v in by_cloud.items()},
            by_industry={k: round(v, 4) for k, v in by_industry.items()},
            by_agent={k: round(v, 4) for k, v in by_agent.items()},
            by_business_unit={k: round(v, 4) for k, v in by_bu.items()},
            token_breakdown={"input": tokens_in, "output": tokens_out, "total": tokens_in + tokens_out},
            top_5_agents=top_5,
            cost_optimisation_tips=tips,
            projected_monthly_usd=round(projected, 2),
        )
