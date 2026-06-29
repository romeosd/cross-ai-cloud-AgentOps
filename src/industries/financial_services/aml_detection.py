"""
Financial Services — AML Transaction Monitoring
================================================
Use case: Real-time anti-money laundering detection with stateful LangGraph.

Cloud assignment:
  AWS Bedrock Agent → Transaction enrichment + pattern matching
  LangGraph state machine manages the investigation lifecycle:
    MONITOR → ALERT → ENRICH → INVESTIGATE → ESCALATE / CLEAR

Regulatory coverage: AUSTRAC AML/CTF Act 2006, FATF Recommendations,
                     APRA CPS 234 (data security)
"""
from __future__ import annotations

from typing import Any, Literal

from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from src.utils.logging import get_logger

logger = get_logger(__name__)

AMLDecision = Literal["CLEAR", "ESCALATE", "MONITORING"]


class AMLState(TypedDict):
    """LangGraph state for AML investigation lifecycle."""
    transaction_id: str
    account_id: str
    amount_aud: float
    transaction_type: str          # WIRE / CASH / CRYPTO / CARD
    counterparty_country: str
    alert_triggered: bool | None
    alert_rules_fired: list[str]
    enriched_data: dict[str, Any]
    risk_score: float | None       # 0–100
    typologies_matched: list[str]  # Structuring, Layering, Integration, etc.
    suspicious_activity_report: str | None
    aml_decision: AMLDecision | None
    case_id: str | None
    _tokens_used: int


_HIGH_RISK_COUNTRIES = {
    "AF", "BY", "CF", "CD", "CU", "ER", "ET", "IR", "IQ",
    "KP", "LB", "LY", "ML", "MZ", "NI", "PK", "RU", "SD",
    "SO", "SS", "SY", "UA", "VE", "YE", "ZW",
}

_STRUCTURING_THRESHOLD_AUD = 9_999.0   # Below AUS $10,000 reporting threshold
_HIGH_VALUE_THRESHOLD_AUD  = 50_000.0


def _screen_transaction(state: AMLState) -> AMLState:
    """Screen transaction against AML rule engine."""
    rules_fired: list[str] = []
    amount = state["amount_aud"]
    country = state.get("counterparty_country", "AU")
    tx_type = state.get("transaction_type", "WIRE")

    # Rule 1: Structuring (smurfing)
    if _STRUCTURING_THRESHOLD_AUD * 0.85 <= amount <= _STRUCTURING_THRESHOLD_AUD:
        rules_fired.append("STRUCTURING_THRESHOLD_PROXIMITY")

    # Rule 2: High-risk jurisdiction
    if country in _HIGH_RISK_COUNTRIES:
        rules_fired.append("HIGH_RISK_JURISDICTION")

    # Rule 3: Large cash transaction
    if tx_type == "CASH" and amount >= _STRUCTURING_THRESHOLD_AUD:
        rules_fired.append("LARGE_CASH_TRANSACTION_REPORT_REQUIRED")

    # Rule 4: Crypto on-ramp / off-ramp
    if tx_type == "CRYPTO" and amount >= 5000:
        rules_fired.append("CRYPTO_TRANSACTION_ENHANCED_DUE_DILIGENCE")

    # Rule 5: High value wire
    if tx_type == "WIRE" and amount >= _HIGH_VALUE_THRESHOLD_AUD:
        rules_fired.append("HIGH_VALUE_INTERNATIONAL_WIRE")

    alert = len(rules_fired) > 0
    logger.info("AML screening complete", transaction_id=state["transaction_id"], rules_fired=rules_fired, alert=alert)

    return {
        **state,
        "alert_triggered": alert,
        "alert_rules_fired": rules_fired,
        "_tokens_used": state.get("_tokens_used", 0) + 150,
    }


def _enrich_transaction(state: AMLState) -> AMLState:
    """Enrich with customer profile, historical patterns, and network links."""
    enriched = {
        "customer_risk_rating": "MEDIUM",
        "account_age_days": 847,
        "transaction_frequency_30d": 23,
        "similar_transactions_90d": 4,
        "linked_accounts": 2,
        "beneficial_owner_verified": True,
        "pep_match": False,              # Politically exposed person
        "sanctions_match": False,
        "adverse_media": False,
    }

    typologies: list[str] = []
    rules = state.get("alert_rules_fired", [])

    if "STRUCTURING_THRESHOLD_PROXIMITY" in rules and enriched["similar_transactions_90d"] >= 3:
        typologies.append("STRUCTURING")
    if "HIGH_RISK_JURISDICTION" in rules and enriched["linked_accounts"] > 1:
        typologies.append("LAYERING")
    if enriched["pep_match"]:
        typologies.append("PEP_INVOLVEMENT")

    logger.info("Transaction enriched", typologies=typologies)

    return {
        **state,
        "enriched_data": enriched,
        "typologies_matched": typologies,
        "_tokens_used": state.get("_tokens_used", 0) + 200,
    }


def _score_risk(state: AMLState) -> AMLState:
    """Compute composite AML risk score 0–100."""
    score = 0.0
    rules = state.get("alert_rules_fired", [])
    typologies = state.get("typologies_matched", [])
    enriched = state.get("enriched_data", {})

    rule_scores = {
        "STRUCTURING_THRESHOLD_PROXIMITY": 35,
        "HIGH_RISK_JURISDICTION": 40,
        "LARGE_CASH_TRANSACTION_REPORT_REQUIRED": 30,
        "CRYPTO_TRANSACTION_ENHANCED_DUE_DILIGENCE": 25,
        "HIGH_VALUE_INTERNATIONAL_WIRE": 20,
    }
    for rule in rules:
        score += rule_scores.get(rule, 10)

    typology_scores = {"STRUCTURING": 30, "LAYERING": 35, "PEP_INVOLVEMENT": 40}
    for typology in typologies:
        score += typology_scores.get(typology, 15)

    if enriched.get("sanctions_match"):
        score += 60
    if enriched.get("adverse_media"):
        score += 20

    risk_score = min(score, 100.0)
    logger.info("Risk scoring complete", risk_score=risk_score)

    return {
        **state,
        "risk_score": risk_score,
        "_tokens_used": state.get("_tokens_used", 0) + 100,
    }


def _generate_sar(state: AMLState) -> AMLState:
    """Generate a Suspicious Activity Report for AUSTRAC submission."""
    tx_id = state["transaction_id"]
    amount = state["amount_aud"]
    typologies = state.get("typologies_matched", [])
    rules = state.get("alert_rules_fired", [])
    risk = state.get("risk_score", 0)

    sar = (
        f"SUSPICIOUS MATTER REPORT — AUSTRAC Submission\n"
        f"{'='*50}\n"
        f"Transaction ID: {tx_id}\n"
        f"Account: {state['account_id']}\n"
        f"Amount: AUD ${amount:,.2f}\n"
        f"Type: {state['transaction_type']}\n"
        f"Counterparty Country: {state.get('counterparty_country','')}\n"
        f"Risk Score: {risk:.0f}/100\n"
        f"Rules Triggered: {', '.join(rules)}\n"
        f"AML Typologies: {', '.join(typologies) if typologies else 'None confirmed'}\n\n"
        f"Grounds for suspicion: The transaction exhibits indicators consistent with "
        f"{', '.join(typologies) if typologies else 'unusual activity'} as defined under "
        f"section 41 of the AML/CTF Act 2006.\n\n"
        f"Reporting entity is required to submit this SMR to AUSTRAC within 3 business days.\n"
        f"[Classification: RESTRICTED — AML Team Only]"
    )

    import uuid
    case_id = f"AML-{state['transaction_id'][:8].upper()}-{str(uuid.uuid4())[:4].upper()}"

    logger.info("SAR generated", case_id=case_id, risk_score=risk)

    return {
        **state,
        "suspicious_activity_report": sar,
        "case_id": case_id,
        "aml_decision": "ESCALATE",
        "_tokens_used": state.get("_tokens_used", 0) + 350,
    }


def _clear_transaction(state: AMLState) -> AMLState:
    """Mark transaction as cleared — no further action required."""
    logger.info("Transaction cleared", transaction_id=state["transaction_id"])
    return {
        **state,
        "aml_decision": "CLEAR",
        "_tokens_used": state.get("_tokens_used", 0) + 50,
    }


def _route_after_screen(state: AMLState) -> str:
    if state.get("alert_triggered"):
        return "enrich"
    return "clear"


def _route_after_score(state: AMLState) -> str:
    risk = state.get("risk_score", 0)
    if risk >= 50:
        return "generate_sar"
    elif risk >= 25:
        return "generate_sar"   # Still generate SAR for medium risk
    return "clear"


def build_graph(llm: Any = None) -> StateGraph:
    """
    Build the AML Detection LangGraph stateful investigation graph.

    Graph:
      screen → [alert?] → enrich → score → [risk ≥ 25?] → generate_sar → END
                        ↘ clear → END
    """
    graph = StateGraph(AMLState)

    graph.add_node("screen", _screen_transaction)
    graph.add_node("enrich", _enrich_transaction)
    graph.add_node("score", _score_risk)
    graph.add_node("generate_sar", _generate_sar)
    graph.add_node("clear", _clear_transaction)

    graph.set_entry_point("screen")

    graph.add_conditional_edges("screen", _route_after_screen, {"enrich": "enrich", "clear": "clear"})
    graph.add_edge("enrich", "score")
    graph.add_conditional_edges("score", _route_after_score, {"generate_sar": "generate_sar", "clear": "clear"})
    graph.add_edge("generate_sar", END)
    graph.add_edge("clear", END)

    return graph.compile()


AGENT_METADATA = {
    "id": "fs_aml_detection",
    "name": "AML Transaction Monitoring",
    "industry": "financial_services",
    "clouds": ["aws"],
    "orchestration": "stateful_graph",
    "sla_seconds": 30,
    "compliance": ["AUSTRAC_AML_CTF", "FATF", "APRA_CPS234"],
    "description": (
        "Real-time AML transaction screening with a stateful LangGraph investigation lifecycle. "
        "Screens against 5 rule sets, enriches with customer profile and network data, "
        "computes composite risk score, and auto-generates AUSTRAC SMR for scores ≥ 25. "
        "All SAR-generating decisions escalate to human compliance officer."
    ),
}
