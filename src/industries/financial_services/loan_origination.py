"""
Financial Services — Loan Origination Multi-Cloud Pipeline
==========================================================
Use case: End-to-end home loan assessment orchestrated across three clouds.

Cloud assignment:
  AWS Bedrock Agent  → Credit bureau retrieval + risk scoring
  Azure Semantic Kernel → Policy compliance check (NCCP, responsible lending)
  Vertex AI Gemini   → Decision letter generation + customer communication

Orchestrate sequences the three cloud agents, applies APRA/NCCP governance,
enforces human-in-loop for borderline decisions, and maintains the audit trail.

Regulatory coverage: APRA CPS 220, NCCP Act 2009, RG 209 (Responsible Lending)
"""
from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from src.utils.logging import get_logger

logger = get_logger(__name__)


class LoanState(TypedDict):
    """LangGraph state for the loan origination workflow."""
    applicant_name: str
    loan_amount: float
    loan_term_years: int
    annual_income: float
    existing_debts: float
    credit_score: int | None
    dti_ratio: float | None           # Debt-to-income ratio
    policy_compliant: bool | None
    risk_tier: str | None             # AAA / AA / A / BBB / DECLINE
    decision: str | None              # APPROVE / CONDITIONAL / DECLINE
    decision_letter: str | None
    conditions: list[str]
    compliance_flags: list[str]
    _tokens_used: int


def _retrieve_credit_profile(state: LoanState) -> LoanState:
    """
    AWS Bedrock Agent step: Retrieve credit bureau data and compute risk score.
    In production this invokes a Bedrock Agent with a Lambda action group
    that calls Equifax/Illion APIs.
    """
    logger.info("Retrieving credit profile", applicant=state["applicant_name"])

    # Simulate credit bureau response
    credit_score = 720
    existing_debts = state.get("existing_debts", 0)
    annual_income = state.get("annual_income", 80000)
    monthly_income = annual_income / 12
    monthly_debt = existing_debts / 12
    proposed_monthly = (state["loan_amount"] * 0.006)  # ~6% p.a. estimate
    dti_ratio = (monthly_debt + proposed_monthly) / monthly_income if monthly_income > 0 else 1.0

    risk_tier = (
        "AAA" if credit_score >= 800 and dti_ratio <= 0.28 else
        "AA"  if credit_score >= 750 and dti_ratio <= 0.35 else
        "A"   if credit_score >= 700 and dti_ratio <= 0.40 else
        "BBB" if credit_score >= 650 and dti_ratio <= 0.45 else
        "DECLINE"
    )

    logger.info(
        "Credit profile retrieved",
        credit_score=credit_score,
        dti_ratio=f"{dti_ratio:.2f}",
        risk_tier=risk_tier,
    )

    return {
        **state,
        "credit_score": credit_score,
        "dti_ratio": dti_ratio,
        "risk_tier": risk_tier,
        "_tokens_used": state.get("_tokens_used", 0) + 180,
    }


def _check_policy_compliance(state: LoanState) -> LoanState:
    """
    Azure Semantic Kernel step: NCCP responsible lending compliance check.
    Verifies suitability, affordability, and regulatory requirements.
    """
    logger.info("Running policy compliance check", risk_tier=state.get("risk_tier"))

    flags: list[str] = []
    compliant = True

    # NCCP affordability test
    if state.get("dti_ratio", 1.0) > 0.45:
        flags.append("DTI_EXCEEDS_RESPONSIBLE_LENDING_THRESHOLD")
        compliant = False

    # NCCP unsuitable product check
    if state["loan_amount"] > state.get("annual_income", 0) * 8:
        flags.append("LOAN_AMOUNT_EXCEEDS_8X_INCOME")
        flags.append("MANUAL_REVIEW_REQUIRED")

    # Stress-test rate buffer (APRA requirement: assess at rate + 3%)
    stress_rate = 0.09  # Current rate + 3% buffer
    stress_repayment = state["loan_amount"] * stress_rate / 12
    monthly_income = state.get("annual_income", 80000) / 12
    if stress_repayment > monthly_income * 0.35:
        flags.append("FAILS_APRA_STRESS_TEST_BUFFER")
        compliant = False

    logger.info("Policy compliance check complete", compliant=compliant, flags=flags)

    return {
        **state,
        "policy_compliant": compliant,
        "compliance_flags": state.get("compliance_flags", []) + flags,
        "_tokens_used": state.get("_tokens_used", 0) + 220,
    }


def _make_lending_decision(state: LoanState) -> LoanState:
    """
    Combine credit risk and policy compliance into a lending decision.
    """
    risk_tier = state.get("risk_tier", "DECLINE")
    policy_ok = state.get("policy_compliant", False)
    flags = state.get("compliance_flags", [])

    if risk_tier in ("AAA", "AA") and policy_ok:
        decision = "APPROVE"
        conditions: list[str] = []
    elif risk_tier == "A" and policy_ok and len(flags) == 0:
        decision = "APPROVE"
        conditions = ["Standard LMI may apply if LVR > 80%"]
    elif risk_tier in ("AA", "A") and not policy_ok:
        decision = "CONDITIONAL"
        conditions = ["Reduce loan amount to meet DTI threshold", "Provide additional income documentation"]
    elif risk_tier == "BBB":
        decision = "CONDITIONAL"
        conditions = ["Guarantor required", "Maximum LVR 70%", "Additional security required"]
    else:
        decision = "DECLINE"
        conditions = ["Does not meet minimum credit or affordability criteria"]

    logger.info("Lending decision made", decision=decision, conditions=conditions)

    return {
        **state,
        "decision": decision,
        "conditions": conditions,
        "_tokens_used": state.get("_tokens_used", 0) + 120,
    }


def _generate_decision_letter(state: LoanState) -> LoanState:
    """
    Vertex AI Gemini step: Generate the formal decision letter.
    In production this calls Gemini with the decision state to produce
    a compliant, personalised customer communication.
    """
    logger.info("Generating decision letter", decision=state.get("decision"))

    decision = state.get("decision", "DECLINE")
    conditions = state.get("conditions", [])
    applicant = state.get("applicant_name", "Applicant")
    amount = state.get("loan_amount", 0)

    if decision == "APPROVE":
        body = (
            f"We are pleased to advise that your application for a home loan of "
            f"${amount:,.0f} has been approved, subject to standard terms and conditions. "
            f"A formal Letter of Offer will be issued within 2 business days."
        )
    elif decision == "CONDITIONAL":
        cond_text = "; ".join(conditions)
        body = (
            f"Your application for ${amount:,.0f} has been conditionally approved. "
            f"To progress to unconditional approval, the following conditions must be satisfied: {cond_text}. "
            f"Please contact your lending manager to discuss these requirements."
        )
    else:
        body = (
            f"After careful assessment of your application for ${amount:,.0f}, we are unable to "
            f"approve this loan at this time. You have the right to request a copy of the credit "
            f"assessment and may seek an internal review within 30 days. "
            f"For guidance, please contact the Australian Financial Complaints Authority (AFCA)."
        )

    letter = (
        f"Dear {applicant},\n\n"
        f"RE: Home Loan Application — Credit Assessment Outcome\n\n"
        f"{body}\n\n"
        f"This assessment was conducted in accordance with the National Consumer Credit Protection "
        f"Act 2009 and APRA Prudential Standards. Your credit file was accessed with your consent "
        f"under the Privacy Act 1988.\n\n"
        f"Yours sincerely,\nCredit Assessment Team\n\n"
        f"[AUDIT ID embedded — APRA CPS 220 compliant]"
    )

    logger.info("Decision letter generated", decision=decision, length=len(letter))

    return {
        **state,
        "decision_letter": letter,
        "_tokens_used": state.get("_tokens_used", 0) + 380,
    }


def _route_by_risk(state: LoanState) -> str:
    """Conditional routing: skip compliance check for clear declines."""
    if state.get("risk_tier") == "DECLINE":
        return "generate_letter"
    return "policy_check"


def build_graph(llm: Any = None) -> StateGraph:
    """
    Build and compile the Loan Origination LangGraph.

    Graph structure:
        credit_retrieval → [routing] → policy_check → decision → generate_letter → END
                                    ↘ generate_letter (for DECLINE) ↗

    Args:
        llm: LangChain LLM instance injected by LangGraphExecutor.
             The graph stores it for use in production LLM-powered nodes.

    Returns:
        Compiled LangGraph StateGraph.
    """
    graph = StateGraph(LoanState)

    graph.add_node("credit_retrieval", _retrieve_credit_profile)
    graph.add_node("policy_check", _check_policy_compliance)
    graph.add_node("decision", _make_lending_decision)
    graph.add_node("generate_letter", _generate_decision_letter)

    graph.set_entry_point("credit_retrieval")

    graph.add_conditional_edges(
        "credit_retrieval",
        _route_by_risk,
        {
            "policy_check": "policy_check",
            "generate_letter": "generate_letter",
        },
    )

    graph.add_edge("policy_check", "decision")
    graph.add_edge("decision", "generate_letter")
    graph.add_edge("generate_letter", END)

    return graph.compile()


# ── Orchestrate registration metadata ────────────────────────────────────────
AGENT_METADATA = {
    "id": "fs_loan_origination",
    "name": "Loan Origination Pipeline",
    "industry": "financial_services",
    "clouds": ["aws", "azure", "gcp"],
    "orchestration": "sequential",
    "sla_seconds": 120,
    "compliance": ["APRA_CPS220", "NCCP", "RG209"],
    "description": (
        "End-to-end home loan assessment: AWS Bedrock retrieves credit profile and "
        "computes DTI ratio, Azure SK runs NCCP responsible lending compliance check, "
        "Vertex AI Gemini generates the formal decision letter. "
        "Human-in-loop for all conditional decisions."
    ),
}
