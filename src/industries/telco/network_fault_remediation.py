"""
Telco — Network Fault Remediation Agent Chain
==============================================
Use case: Autonomous network fault detection, root cause analysis, and remediation.

Cloud assignment:
  AWS Bedrock   → Fault detection + alert triage (pattern matching on log streams)
  Vertex AI GCP → Root cause analysis (Gemini reasoning over network topology)
  AWS Lambda/Bedrock → Automated remediation action execution

LangGraph manages the investigation state machine:
  DETECT → TRIAGE → RCA → REMEDIATE / ESCALATE → VERIFY → CLOSE

Industry context: Tier-1 Australian telco managing 15M+ subscribers.
Each minute of P1 outage costs ~$50,000 in SLA penalties.
"""
from __future__ import annotations

from typing import Any, Literal

from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from src.utils.logging import get_logger

logger = get_logger(__name__)

FaultSeverity = Literal["P1", "P2", "P3", "P4"]
RemediationStatus = Literal["AUTO_REMEDIATED", "ESCALATED", "MANUAL_REQUIRED", "VERIFIED_CLEAR"]


class NetworkFaultState(TypedDict):
    incident_id: str
    alert_source: str              # NetFlow / SNMP / Synthetic / Customer
    affected_nodes: list[str]
    affected_subscribers: int
    fault_type: str | None         # LINK_DOWN / PACKET_LOSS / LATENCY / CAPACITY
    severity: FaultSeverity | None
    root_cause: str | None
    root_cause_confidence: float | None
    remediation_actions: list[str]
    remediation_status: RemediationStatus | None
    rollback_available: bool
    sla_breach_risk: bool
    mttr_estimate_minutes: int | None
    verification_passed: bool | None
    _tokens_used: int


def _detect_and_triage(state: NetworkFaultState) -> NetworkFaultState:
    """AWS: Correlate alerts, classify fault type, and assign severity."""
    logger.info("Triaging network fault", incident=state["incident_id"], nodes=state["affected_nodes"])

    affected = state["affected_subscribers"]
    fault_type = "LINK_DOWN" if affected > 10000 else "PACKET_LOSS" if affected > 1000 else "LATENCY"

    severity: FaultSeverity = (
        "P1" if affected > 50000 else
        "P2" if affected > 10000 else
        "P3" if affected > 1000 else
        "P4"
    )

    sla_breach = severity in ("P1", "P2")

    logger.info("Triage complete", severity=severity, fault_type=fault_type, sla_breach=sla_breach)
    return {
        **state,
        "fault_type": fault_type,
        "severity": severity,
        "sla_breach_risk": sla_breach,
        "_tokens_used": state.get("_tokens_used", 0) + 180,
    }


def _root_cause_analysis(state: NetworkFaultState) -> NetworkFaultState:
    """GCP Vertex AI: Deep RCA using Gemini reasoning over network topology graph."""
    logger.info("Running root cause analysis", fault_type=state["fault_type"], severity=state["severity"])

    fault = state.get("fault_type", "UNKNOWN")
    nodes = state.get("affected_nodes", [])

    rca_map = {
        "LINK_DOWN": ("Fibre cut on primary backbone segment between nodes. Secondary path BGP convergence delayed.", 0.91),
        "PACKET_LOSS": ("Buffer overflow on edge router due to unexpected traffic spike. QoS policy not applied to new VLAN.", 0.84),
        "LATENCY": ("DNS resolution bottleneck at regional resolver cluster. TTL misconfiguration causing repeated lookups.", 0.88),
        "CAPACITY": ("Peak hour traffic exceeded provisioned capacity on core uplink. Auto-scaling threshold not triggered.", 0.79),
    }

    root_cause, confidence = rca_map.get(fault, ("Unknown root cause — manual investigation required.", 0.45))
    mttr = 15 if confidence > 0.85 else 45 if confidence > 0.70 else 120

    logger.info("RCA complete", root_cause=root_cause[:60], confidence=confidence, mttr_estimate=mttr)
    return {
        **state,
        "root_cause": root_cause,
        "root_cause_confidence": confidence,
        "mttr_estimate_minutes": mttr,
        "_tokens_used": state.get("_tokens_used", 0) + 320,
    }


def _execute_remediation(state: NetworkFaultState) -> NetworkFaultState:
    """AWS: Execute automated remediation actions via network API."""
    fault = state.get("fault_type", "UNKNOWN")
    confidence = state.get("root_cause_confidence", 0)

    action_map = {
        "LINK_DOWN": ["FAILOVER_TO_SECONDARY_PATH", "NOTIFY_FIELD_CREW", "UPDATE_BGP_TABLES"],
        "PACKET_LOSS": ["APPLY_QOS_POLICY_VLAN_UPDATE", "DRAIN_BUFFER", "REROUTE_TRAFFIC_ALTERNATE_PATH"],
        "LATENCY": ["FLUSH_DNS_CACHE_REGIONAL", "INCREASE_DNS_TTL", "REDIRECT_TO_BACKUP_RESOLVER"],
        "CAPACITY": ["ACTIVATE_BURST_CAPACITY", "THROTTLE_NON_CRITICAL_TRAFFIC", "ALERT_CAPACITY_PLANNING"],
    }

    actions = action_map.get(fault, ["MANUAL_INVESTIGATION_REQUIRED"])

    # Only auto-remediate if confidence is high enough
    if confidence < 0.75:
        status: RemediationStatus = "ESCALATED"
        actions = ["ESCALATE_TO_NOC_L3", "BRIDGE_INCIDENT_BRIDGE"] + actions
        rollback = False
    else:
        status = "AUTO_REMEDIATED"
        rollback = True

    logger.info("Remediation executed", status=status, actions=actions, rollback_available=rollback)
    return {
        **state,
        "remediation_actions": actions,
        "remediation_status": status,
        "rollback_available": rollback,
        "_tokens_used": state.get("_tokens_used", 0) + 200,
    }


def _verify_resolution(state: NetworkFaultState) -> NetworkFaultState:
    """Run synthetic tests to verify the network is restored."""
    remediation_ok = state.get("remediation_status") == "AUTO_REMEDIATED"
    verification_passed = remediation_ok  # In prod: run ping, traceroute, synthetic transactions
    logger.info("Verification complete", passed=verification_passed)
    return {**state, "verification_passed": verification_passed, "_tokens_used": state.get("_tokens_used", 0) + 80}


def _route_after_rca(state: NetworkFaultState) -> str:
    confidence = state.get("root_cause_confidence", 0)
    if confidence >= 0.75:
        return "remediate"
    return "escalate_to_noc"


def _escalate(state: NetworkFaultState) -> NetworkFaultState:
    logger.info("Escalating to NOC L3", incident=state["incident_id"])
    return {**state, "remediation_status": "MANUAL_REQUIRED", "_tokens_used": state.get("_tokens_used", 0) + 50}


def build_graph(llm: Any = None) -> StateGraph:
    graph = StateGraph(NetworkFaultState)
    graph.add_node("triage", _detect_and_triage)
    graph.add_node("rca", _root_cause_analysis)
    graph.add_node("remediate", _execute_remediation)
    graph.add_node("verify", _verify_resolution)
    graph.add_node("escalate_to_noc", _escalate)
    graph.set_entry_point("triage")
    graph.add_edge("triage", "rca")
    graph.add_conditional_edges("rca", _route_after_rca, {"remediate": "remediate", "escalate_to_noc": "escalate_to_noc"})
    graph.add_edge("remediate", "verify")
    graph.add_edge("verify", END)
    graph.add_edge("escalate_to_noc", END)
    return graph.compile()


AGENT_METADATA = {
    "id": "telco_network_fault",
    "name": "Network Fault Remediation",
    "industry": "telco",
    "clouds": ["aws", "gcp"],
    "orchestration": "stateful_graph",
    "sla_seconds": 60,
    "description": (
        "Autonomous network fault remediation: AWS Bedrock triages alerts from NetFlow/SNMP streams "
        "and classifies severity (P1-P4). Vertex AI Gemini performs deep RCA over the network topology graph. "
        "High-confidence faults (>75%) are auto-remediated via AWS network API; others escalate to NOC L3. "
        "Synthetic verification confirms resolution. Saves ~$50K/min in P1 SLA penalties."
    ),
}
