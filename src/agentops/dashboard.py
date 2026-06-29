"""
AgentOps Dashboard — Single Pane of Glass for all cloud agents.

Aggregates real-time metrics from all registered agents across
AWS Bedrock, Azure Semantic Kernel, and Vertex AI into one view.

Exposes:
- FastAPI REST endpoint at /metrics and /dashboard/summary
- Prometheus metrics endpoint at /metrics/prometheus
- Per-agent health, cost, latency, success rate
- Per-cloud cost breakdown and token usage
- Per-industry usage heatmap
- Governance escalation stats
- Real-time alert feed
"""
from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from src.control_plane.agent_registry import AgentRegistry, AgentStatus
from src.utils.config import get_config
from src.utils.logging import get_logger

logger = get_logger(__name__)

# ── Prometheus metrics ────────────────────────────────────────────────────────
AGENT_INVOCATIONS = Counter(
    "agentops_invocations_total",
    "Total agent invocations",
    ["agent_id", "cloud", "industry", "status"],
)
AGENT_LATENCY = Histogram(
    "agentops_latency_seconds",
    "Agent invocation latency",
    ["agent_id", "cloud"],
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0],
)
AGENT_COST = Counter(
    "agentops_cost_usd_total",
    "Cumulative agent cost in USD",
    ["agent_id", "cloud", "industry"],
)
AGENT_TOKENS = Counter(
    "agentops_tokens_total",
    "Total tokens consumed",
    ["agent_id", "cloud"],
)
ESCALATIONS = Counter(
    "agentops_escalations_total",
    "Human escalations triggered",
    ["agent_id", "reason"],
)
CLOUD_HEALTH = Gauge(
    "agentops_cloud_health",
    "Cloud execution plane health (1=healthy, 0=degraded)",
    ["cloud"],
)


@dataclass
class AgentMetrics:
    """Aggregated metrics for a single agent."""
    agent_id: str
    industry: str
    clouds: list[str]
    total_invocations: int = 0
    successful_invocations: int = 0
    failed_invocations: int = 0
    escalated_invocations: int = 0
    total_cost_usd: float = 0.0
    total_tokens: int = 0
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    last_invoked: float = 0.0
    status: str = "active"

    @property
    def success_rate(self) -> float:
        return self.successful_invocations / self.total_invocations if self.total_invocations else 1.0

    @property
    def escalation_rate(self) -> float:
        return self.escalated_invocations / self.total_invocations if self.total_invocations else 0.0


@dataclass
class CloudMetrics:
    """Aggregated metrics per cloud provider."""
    cloud: str
    total_invocations: int = 0
    total_cost_usd: float = 0.0
    total_tokens: int = 0
    avg_latency_ms: float = 0.0
    health: float = 1.0


@dataclass
class DashboardSummary:
    """Complete AgentOps dashboard snapshot."""
    timestamp: float
    total_agents_registered: int
    agents_active: int
    agents_degraded: int
    agents_offline: int
    total_invocations_24h: int
    total_cost_usd_24h: float
    total_tokens_24h: int
    avg_latency_ms_24h: float
    overall_success_rate: float
    escalation_rate: float
    by_industry: dict[str, dict[str, Any]]
    by_cloud: dict[str, CloudMetrics]
    top_cost_agents: list[dict[str, Any]]
    recent_escalations: list[dict[str, Any]]
    alerts: list[dict[str, Any]]


class AgentOpsDashboard:
    """
    Single pane of glass for all AI agents across AWS, Azure, and GCP.

    Ingests invocation telemetry from OrchestrateClient and surfaces
    it via REST API and Prometheus metrics.

    Example:
        dashboard = AgentOpsDashboard()

        # Record an invocation
        dashboard.record_invocation(
            agent_id="fs_loan_origination",
            cloud="aws",
            industry="financial_services",
            latency_ms=1240,
            cost_usd=0.042,
            tokens=1400,
            success=True,
            escalated=False,
        )

        # Get summary
        summary = dashboard.get_summary()
        print(f"Total 24h cost: ${summary.total_cost_usd_24h:.2f}")
        print(f"Success rate: {summary.overall_success_rate:.1%}")
    """

    def __init__(self) -> None:
        self._registry = AgentRegistry()
        self._cfg = get_config()

        # In-memory metrics store (production: Prometheus + Grafana / CloudWatch)
        self._agent_metrics: dict[str, AgentMetrics] = {}
        self._cloud_metrics: dict[str, CloudMetrics] = {
            "aws": CloudMetrics(cloud="aws"),
            "azure": CloudMetrics(cloud="azure"),
            "gcp": CloudMetrics(cloud="gcp"),
        }
        self._invocation_history: list[dict[str, Any]] = []
        self._escalation_history: list[dict[str, Any]] = []
        self._alerts: list[dict[str, Any]] = []

        # Initialise agent metrics from registry
        for agent in self._registry.list_all():
            self._agent_metrics[agent.id] = AgentMetrics(
                agent_id=agent.id,
                industry=agent.industry,
                clouds=agent.clouds,
            )

        # Set cloud health to 1 (healthy) on startup
        for cloud in ("aws", "azure", "gcp"):
            CLOUD_HEALTH.labels(cloud=cloud).set(1)

        logger.info("AgentOpsDashboard initialised", agents=len(self._agent_metrics))

    def record_invocation(
        self,
        agent_id: str,
        cloud: str,
        industry: str,
        latency_ms: float,
        cost_usd: float,
        tokens: int,
        success: bool,
        escalated: bool = False,
        escalation_reason: str = "",
    ) -> None:
        """Record a single agent invocation for metrics aggregation."""
        status = "success" if success else "failure"

        # Prometheus
        AGENT_INVOCATIONS.labels(agent_id=agent_id, cloud=cloud, industry=industry, status=status).inc()
        AGENT_LATENCY.labels(agent_id=agent_id, cloud=cloud).observe(latency_ms / 1000)
        AGENT_COST.labels(agent_id=agent_id, cloud=cloud, industry=industry).inc(cost_usd)
        AGENT_TOKENS.labels(agent_id=agent_id, cloud=cloud).inc(tokens)
        if escalated:
            ESCALATIONS.labels(agent_id=agent_id, reason=escalation_reason).inc()

        # In-memory aggregation
        if agent_id in self._agent_metrics:
            m = self._agent_metrics[agent_id]
            m.total_invocations += 1
            if success:
                m.successful_invocations += 1
            else:
                m.failed_invocations += 1
            if escalated:
                m.escalated_invocations += 1
            m.total_cost_usd += cost_usd
            m.total_tokens += tokens
            m.avg_latency_ms = (m.avg_latency_ms * (m.total_invocations - 1) + latency_ms) / m.total_invocations
            m.last_invoked = time.time()

        # Cloud metrics
        if cloud in self._cloud_metrics:
            cm = self._cloud_metrics[cloud]
            cm.total_invocations += 1
            cm.total_cost_usd += cost_usd
            cm.total_tokens += tokens
            cm.avg_latency_ms = (cm.avg_latency_ms * (cm.total_invocations - 1) + latency_ms) / cm.total_invocations

        # History
        record = {
            "timestamp": time.time(),
            "agent_id": agent_id,
            "cloud": cloud,
            "industry": industry,
            "latency_ms": latency_ms,
            "cost_usd": cost_usd,
            "tokens": tokens,
            "success": success,
            "escalated": escalated,
        }
        self._invocation_history.append(record)
        if len(self._invocation_history) > 10000:
            self._invocation_history = self._invocation_history[-5000:]

        if escalated:
            self._escalation_history.append({**record, "reason": escalation_reason})

        # Cost alerts
        daily_cost = sum(r["cost_usd"] for r in self._invocation_history if r["timestamp"] > time.time() - 86400)
        threshold = self._cfg.agentops.get("cost_alert_threshold_usd", 100.0)
        if daily_cost > threshold and not any(a["type"] == "COST_THRESHOLD" for a in self._alerts[-5:]):
            self._alerts.append({
                "type": "COST_THRESHOLD",
                "message": f"Daily agent cost ${daily_cost:.2f} exceeds threshold ${threshold:.2f}",
                "severity": "WARNING",
                "timestamp": time.time(),
            })
            logger.warning("Cost threshold exceeded", daily_cost=daily_cost, threshold=threshold)

    def get_summary(self, window_hours: int = 24) -> DashboardSummary:
        """Get a full dashboard summary for the specified time window."""
        cutoff = time.time() - (window_hours * 3600)
        window_records = [r for r in self._invocation_history if r["timestamp"] >= cutoff]

        total = len(window_records)
        successful = sum(1 for r in window_records if r["success"])
        escalated_count = sum(1 for r in window_records if r["escalated"])
        total_cost = sum(r["cost_usd"] for r in window_records)
        total_tokens = sum(r["tokens"] for r in window_records)
        avg_latency = sum(r["latency_ms"] for r in window_records) / total if total else 0

        # By industry
        by_industry: dict[str, dict[str, Any]] = defaultdict(lambda: {"invocations": 0, "cost_usd": 0.0, "success_rate": 1.0})
        industry_success: dict[str, list[bool]] = defaultdict(list)
        for r in window_records:
            ind = r["industry"]
            by_industry[ind]["invocations"] += 1
            by_industry[ind]["cost_usd"] += r["cost_usd"]
            industry_success[ind].append(r["success"])
        for ind in by_industry:
            successes = industry_success[ind]
            by_industry[ind]["success_rate"] = sum(successes) / len(successes) if successes else 1.0

        # Top cost agents
        agent_costs: dict[str, float] = defaultdict(float)
        for r in window_records:
            agent_costs[r["agent_id"]] += r["cost_usd"]
        top_cost = sorted(
            [{"agent_id": k, "cost_usd": v} for k, v in agent_costs.items()],
            key=lambda x: x["cost_usd"],
            reverse=True,
        )[:5]

        registry_summary = self._registry.summary()

        return DashboardSummary(
            timestamp=time.time(),
            total_agents_registered=registry_summary["total_agents"],
            agents_active=registry_summary["active"],
            agents_degraded=registry_summary["degraded"],
            agents_offline=registry_summary["offline"],
            total_invocations_24h=total,
            total_cost_usd_24h=total_cost,
            total_tokens_24h=total_tokens,
            avg_latency_ms_24h=avg_latency,
            overall_success_rate=successful / total if total else 1.0,
            escalation_rate=escalated_count / total if total else 0.0,
            by_industry=dict(by_industry),
            by_cloud=self._cloud_metrics,
            top_cost_agents=top_cost,
            recent_escalations=self._escalation_history[-10:],
            alerts=self._alerts[-20:],
        )

    def get_agent_detail(self, agent_id: str) -> dict[str, Any]:
        """Return detailed metrics for a single agent."""
        agent = self._registry.get(agent_id)
        metrics = self._agent_metrics.get(agent_id)

        return {
            "agent": agent.to_skill_card(),
            "metrics": {
                "total_invocations": metrics.total_invocations if metrics else 0,
                "success_rate": metrics.success_rate if metrics else 1.0,
                "escalation_rate": metrics.escalation_rate if metrics else 0.0,
                "total_cost_usd": metrics.total_cost_usd if metrics else 0.0,
                "avg_latency_ms": metrics.avg_latency_ms if metrics else 0.0,
                "total_tokens": metrics.total_tokens if metrics else 0,
            },
        }


# ── FastAPI application ───────────────────────────────────────────────────────

dashboard = AgentOpsDashboard()
app = FastAPI(
    title="AgentOps Control Plane Dashboard",
    description="Single pane of glass for all AI agents across AWS, Azure, and GCP",
    version="1.0.0",
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "healthy", "service": "AgentOps Dashboard"}


@app.get("/dashboard/summary")
async def get_dashboard_summary(window_hours: int = 24) -> JSONResponse:
    summary = dashboard.get_summary(window_hours=window_hours)
    return JSONResponse(content={
        "timestamp": summary.timestamp,
        "window_hours": window_hours,
        "registered_agents": summary.total_agents_registered,
        "active_agents": summary.agents_active,
        "invocations_24h": summary.total_invocations_24h,
        "total_cost_usd_24h": round(summary.total_cost_usd_24h, 4),
        "total_tokens_24h": summary.total_tokens_24h,
        "avg_latency_ms": round(summary.avg_latency_ms_24h, 1),
        "success_rate": round(summary.overall_success_rate, 4),
        "escalation_rate": round(summary.escalation_rate, 4),
        "by_industry": {k: {**v, "cost_usd": round(v["cost_usd"], 4)} for k, v in summary.by_industry.items()},
        "by_cloud": {k: {"invocations": v.total_invocations, "cost_usd": round(v.total_cost_usd, 4)} for k, v in summary.by_cloud.items()},
        "top_cost_agents": summary.top_cost_agents,
        "recent_escalations": summary.recent_escalations[-5:],
        "alerts": summary.alerts[-5:],
    })


@app.get("/dashboard/agents/{agent_id}")
async def get_agent_detail(agent_id: str) -> JSONResponse:
    detail = dashboard.get_agent_detail(agent_id)
    return JSONResponse(content=detail)


@app.get("/dashboard/registry")
async def get_registry(industry: str | None = None) -> JSONResponse:
    agents = dashboard._registry.list_by_industry(industry) if industry else dashboard._registry.list_all()
    return JSONResponse(content={"agents": [a.to_skill_card() for a in agents], "count": len(agents)})


@app.get("/metrics/prometheus")
async def prometheus_metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
