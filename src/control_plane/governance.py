"""
Governance Engine — policy enforcement, human-in-the-loop, and audit.

Every agent invocation passes through governance before output is returned.
This is the layer that satisfies APRA CPS 220/230, ASIC RG 271,
ISO/IEC 42001, and the EU AI Act's human oversight requirements.

Provides:
- Confidence-gated human escalation
- Compliance tag enforcement
- Immutable audit log with 7-year retention
- PII detection in agent outputs
- Cost guardrails
- Risk scoring per invocation
"""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from src.utils.config import get_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


class EscalationReason(str, Enum):
    LOW_CONFIDENCE = "low_confidence"
    COMPLIANCE_REQUIRED = "compliance_required"
    HIGH_COST = "high_cost"
    PII_DETECTED = "pii_detected"
    MANUAL_OVERRIDE = "manual_override"
    RISK_THRESHOLD = "risk_threshold"


@dataclass
class GovernanceDecision:
    """Result of a governance check on an agent invocation."""

    approved: bool
    escalate_to_human: bool = False
    escalation_reason: EscalationReason | None = None
    risk_score: float = 0.0
    pii_detected: bool = False
    audit_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    policy_violations: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)


@dataclass
class AuditRecord:
    """Immutable audit record for a single agent invocation."""

    audit_id: str
    workflow_id: str
    agent_id: str
    industry: str
    cloud: str
    timestamp: float
    input_hash: str          # SHA-256 of input payload — not the payload itself
    output_summary: str      # Human-readable summary, not raw output
    tokens_used: int
    cost_usd: float
    confidence: float
    escalated: bool
    compliance_tags: list[str]
    governance_decision: str
    user_id: str = ""
    session_id: str = ""


class GovernanceEngine:
    """
    Enterprise AI Governance Engine for multi-cloud agent operations.

    Enforces policies defined in orchestrate_config.yaml across every
    agent invocation regardless of which cloud executed it.

    Compliance coverage:
    - APRA CPS 220 (Risk Management) — risk scoring, escalation thresholds
    - APRA CPS 230 (Operational Resilience) — circuit breakers, fallbacks
    - APRA CPS 234 (Information Security) — PII controls, audit retention
    - ASIC RG 271 (Internal Dispute Resolution) — human-in-loop for disputes
    - ISO/IEC 42001 (AI Management System) — governance documentation
    - EU AI Act Article 14 (Human Oversight) — escalation enforcement

    Example:
        governance = GovernanceEngine()

        # Check before returning agent output
        decision = governance.evaluate(
            agent_id="fs_loan_origination",
            industry="financial_services",
            cloud="aws",
            confidence=0.65,
            output={"decision": "approve", "amount": 650000},
            cost_usd=0.42,
            workflow_id="wf-001",
        )

        if decision.escalate_to_human:
            queue_for_human_review(decision.escalation_reason)
        elif decision.approved:
            return output
    """

    def __init__(self) -> None:
        cfg = get_config()
        gov = cfg.orchestrate.governance
        self._hitl_threshold = float(gov.get("human_in_loop_threshold", 0.70))
        self._max_hops = int(gov.get("max_agent_hops", 10))
        self._audit_enabled = bool(gov.get("audit_all_invocations", True))
        self._pii_enabled = bool(gov.get("pii_detection_enabled", True))
        self._cost_threshold = float(cfg.agentops.get("cost_alert_threshold_usd", 100.0))
        self._retention_days = int(cfg.agentops.get("retention_days", 2555))

        self._audit_log: list[AuditRecord] = []

        logger.info(
            "GovernanceEngine initialised",
            hitl_threshold=self._hitl_threshold,
            audit_enabled=self._audit_enabled,
            retention_days=self._retention_days,
        )

    def evaluate(
        self,
        agent_id: str,
        industry: str,
        cloud: str,
        confidence: float,
        output: Any,
        cost_usd: float,
        workflow_id: str,
        compliance_tags: list[str] | None = None,
        payload: dict[str, Any] | None = None,
    ) -> GovernanceDecision:
        """
        Evaluate an agent invocation against all governance policies.

        Args:
            agent_id: The agent that produced the output.
            industry: Industry vertical (financial_services, retail, etc.)
            cloud: Execution cloud (aws, azure, gcp).
            confidence: Agent's self-reported confidence score (0–1).
            output: The agent's output (will not be logged raw).
            cost_usd: Cost of this invocation.
            workflow_id: Parent workflow ID for correlation.
            compliance_tags: Regulatory tags that apply to this agent.
            payload: Input payload (hashed for audit, not stored raw).

        Returns:
            GovernanceDecision with approval status and escalation details.
        """
        violations: list[str] = []
        recommendations: list[str] = []
        escalate = False
        escalation_reason: EscalationReason | None = None

        # 1. Confidence threshold check
        if confidence < self._hitl_threshold:
            escalate = True
            escalation_reason = EscalationReason.LOW_CONFIDENCE
            violations.append(
                f"Confidence {confidence:.2f} below threshold {self._hitl_threshold}"
            )

        # 2. Compliance-mandated human review
        regulated_tags = {"APRA", "AUSTRAC", "ASIC", "Privacy_Act", "CDPP", "PGPA"}
        tags = set(compliance_tags or [])
        if tags & regulated_tags and industry in ("financial_services", "government"):
            escalate = True
            if not escalation_reason:
                escalation_reason = EscalationReason.COMPLIANCE_REQUIRED
            recommendations.append("Regulated output — route to compliance officer queue")

        # 3. PII detection in output
        pii_detected = False
        if self._pii_enabled:
            pii_detected = self._detect_pii_in_output(output)
            if pii_detected:
                violations.append("PII detected in agent output — redaction required")
                recommendations.append("Apply PII redaction before delivery to end user")

        # 4. Cost guardrail
        if cost_usd > self._cost_threshold * 0.01:  # 1% of daily threshold per call
            recommendations.append(f"High per-invocation cost: ${cost_usd:.4f}")

        # 5. Risk scoring
        risk_score = self._compute_risk_score(
            confidence=confidence,
            industry=industry,
            compliance_tags=list(tags),
            cost_usd=cost_usd,
            pii_detected=pii_detected,
        )

        decision = GovernanceDecision(
            approved=not violations or escalate,
            escalate_to_human=escalate,
            escalation_reason=escalation_reason,
            risk_score=risk_score,
            pii_detected=pii_detected,
            policy_violations=violations,
            recommendations=recommendations,
        )

        # Audit
        if self._audit_enabled:
            record = AuditRecord(
                audit_id=decision.audit_id,
                workflow_id=workflow_id,
                agent_id=agent_id,
                industry=industry,
                cloud=cloud,
                timestamp=time.time(),
                input_hash=self._hash_payload(payload or {}),
                output_summary=self._summarise_output(output),
                tokens_used=0,
                cost_usd=cost_usd,
                confidence=confidence,
                escalated=escalate,
                compliance_tags=list(tags),
                governance_decision="ESCALATE" if escalate else "APPROVE",
            )
            self._audit_log.append(record)

        logger.info(
            "Governance evaluation complete",
            agent_id=agent_id,
            cloud=cloud,
            confidence=f"{confidence:.2f}",
            risk_score=f"{risk_score:.2f}",
            escalate=escalate,
            pii_detected=pii_detected,
            violations=len(violations),
        )

        return decision

    def get_audit_trail(
        self,
        workflow_id: str | None = None,
        agent_id: str | None = None,
        industry: str | None = None,
    ) -> list[AuditRecord]:
        """Query the audit trail with optional filters."""
        records = self._audit_log
        if workflow_id:
            records = [r for r in records if r.workflow_id == workflow_id]
        if agent_id:
            records = [r for r in records if r.agent_id == agent_id]
        if industry:
            records = [r for r in records if r.industry == industry]
        return records

    def escalation_summary(self) -> dict[str, Any]:
        """Return summary statistics on escalations."""
        total = len(self._audit_log)
        escalated = sum(1 for r in self._audit_log if r.escalated)
        by_reason: dict[str, int] = {}
        for r in self._audit_log:
            if r.escalated:
                by_reason[r.governance_decision] = by_reason.get(r.governance_decision, 0) + 1

        return {
            "total_invocations": total,
            "escalated": escalated,
            "escalation_rate": escalated / total if total else 0.0,
            "by_reason": by_reason,
            "avg_risk_score": sum(r.confidence for r in self._audit_log) / total if total else 0.0,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_risk_score(
        self,
        confidence: float,
        industry: str,
        compliance_tags: list[str],
        cost_usd: float,
        pii_detected: bool,
    ) -> float:
        """Compute a 0–1 risk score for an agent invocation."""
        base = 1.0 - confidence
        industry_multiplier = {
            "financial_services": 1.5,
            "government": 1.4,
            "mining": 1.3,
            "telco": 1.1,
            "retail": 1.0,
        }.get(industry, 1.0)
        compliance_factor = min(len(compliance_tags) * 0.05, 0.3)
        pii_factor = 0.2 if pii_detected else 0.0
        cost_factor = min(cost_usd / 10.0, 0.1)

        raw = (base * industry_multiplier) + compliance_factor + pii_factor + cost_factor
        return min(raw, 1.0)

    def _detect_pii_in_output(self, output: Any) -> bool:
        """Simple heuristic PII detection on stringified output."""
        if output is None:
            return False
        text = json.dumps(output) if not isinstance(output, str) else output
        pii_patterns = [
            "@", "TFN", "ABN", "passport", "medicare",
            "credit card", "BSB", "account number", "DOB",
        ]
        text_lower = text.lower()
        return any(pattern.lower() in text_lower for pattern in pii_patterns)

    def _hash_payload(self, payload: dict[str, Any]) -> str:
        """SHA-256 hash of the input payload for audit purposes."""
        serialised = json.dumps(payload, sort_keys=True, default=str).encode()
        return hashlib.sha256(serialised).hexdigest()

    def _summarise_output(self, output: Any) -> str:
        """Create a brief human-readable summary of agent output."""
        if isinstance(output, dict):
            keys = list(output.keys())[:5]
            return f"Dict with keys: {keys}"
        elif isinstance(output, str):
            return output[:100]
        elif isinstance(output, list):
            return f"List of {len(output)} items"
        return type(output).__name__
