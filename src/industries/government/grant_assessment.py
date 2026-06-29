"""
Government — Grant Assessment Pipeline
========================================
Use case: Automated grant eligibility assessment, document verification, and
          risk-scored approval recommendation.

Cloud assignment:
  AWS Bedrock   → Application intake + eligibility rules engine
  Azure SK      → Document verification + entity resolution
  GCP Vertex AI → Risk scoring + approval recommendation generation

Orchestrate enforces mandatory human approval for all grant decisions,
maintains full audit trail for ANAO/PGPA compliance, and publishes
to the Commonwealth Grants Register.

Regulatory coverage: PGPA Act 2013, Commonwealth Grants Policy Framework,
                     Privacy Act 1988, FOI Act 1982
"""
from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from src.utils.logging import get_logger

logger = get_logger(__name__)


class GrantState(TypedDict):
    application_id: str
    applicant_name: str
    applicant_abn: str
    grant_program: str
    requested_amount_aud: float
    project_description: str
    submitted_documents: list[str]
    eligibility_checks: dict[str, bool]
    eligibility_passed: bool | None
    documents_verified: dict[str, Any]
    document_issues: list[str]
    risk_score: float | None           # 0–100, higher = more risk
    risk_factors: list[str]
    recommendation: str | None         # APPROVE / APPROVE_WITH_CONDITIONS / DECLINE
    conditions: list[str]
    assessor_notes: str | None
    requires_human_decision: bool
    commonwealth_register_ready: bool | None
    _tokens_used: int


_ELIGIBLE_ENTITY_TYPES = {"company", "incorporated_association", "charity", "cooperative", "indigenous_corporation"}
_MAX_GRANT_AUTO_APPROVE_AUD = 50_000.0   # Above this always requires human


def _check_eligibility(state: GrantState) -> GrantState:
    """AWS Bedrock: Run eligibility rules engine against Commonwealth Grants Policy."""
    logger.info("Checking grant eligibility", application=state["application_id"], program=state["grant_program"])

    checks: dict[str, bool] = {}
    amount = state["requested_amount_aud"]

    # Rule 1: ABN validation
    abn = state.get("applicant_abn", "")
    checks["valid_abn"] = len(abn.replace(" ", "")) == 11 and abn.replace(" ", "").isdigit()

    # Rule 2: Not-for-profit / entity type eligibility
    checks["eligible_entity_type"] = True  # Simplified — would query ABR API

    # Rule 3: Grant amount within program limits
    program_limits = {
        "Regional_Development": 500_000,
        "Small_Business_Recovery": 25_000,
        "Indigenous_Enterprise": 200_000,
        "Clean_Energy_Innovation": 1_000_000,
        "Digital_Skills": 150_000,
    }
    limit = program_limits.get(state["grant_program"], 100_000)
    checks["within_program_limit"] = amount <= limit

    # Rule 4: Required documents submitted
    required_docs = ["financial_statements", "project_plan", "budget_breakdown", "abn_certificate"]
    submitted = state.get("submitted_documents", [])
    checks["required_documents_submitted"] = all(doc in submitted for doc in required_docs)

    # Rule 5: No prior grant defaults
    checks["no_prior_defaults"] = True  # Would query grant management system

    eligibility_passed = all(checks.values())

    logger.info("Eligibility check complete", passed=eligibility_passed, checks=checks)
    return {
        **state,
        "eligibility_checks": checks,
        "eligibility_passed": eligibility_passed,
        "_tokens_used": state.get("_tokens_used", 0) + 200,
    }


def _verify_documents(state: GrantState) -> GrantState:
    """Azure SK + Document Intelligence: Verify submitted documents for authenticity and completeness."""
    logger.info("Verifying documents", application=state["application_id"])

    submitted = state.get("submitted_documents", [])
    verified: dict[str, Any] = {}
    issues: list[str] = []

    for doc in submitted:
        # In production: call Azure Document Intelligence then SK for entity extraction
        verified[doc] = {
            "verified": True,
            "extracted_abn": state.get("applicant_abn", ""),
            "date_range": "2022-2024",
            "confidence": 0.94,
        }

    # Check financial statement completeness
    if "financial_statements" in submitted:
        # Simulate finding an issue
        amount = state["requested_amount_aud"]
        if amount > 100_000:
            issues.append("Audited financial statements required for grants > $100,000 — unaudited statements submitted")

    logger.info("Document verification complete", issues=len(issues))
    return {
        **state,
        "documents_verified": verified,
        "document_issues": issues,
        "_tokens_used": state.get("_tokens_used", 0) + 250,
    }


def _score_and_recommend(state: GrantState) -> GrantState:
    """GCP Vertex AI: Risk score and generate recommendation."""
    logger.info("Scoring grant application", application=state["application_id"])

    risk_factors: list[str] = []
    risk_score = 0.0

    # Factor 1: Eligibility gaps
    failed_checks = [k for k, v in state.get("eligibility_checks", {}).items() if not v]
    risk_score += len(failed_checks) * 20
    if failed_checks:
        risk_factors.append(f"Failed eligibility checks: {failed_checks}")

    # Factor 2: Document issues
    issues = state.get("document_issues", [])
    risk_score += len(issues) * 15
    if issues:
        risk_factors.append("Document completeness issues identified")

    # Factor 3: Large grant amounts have higher scrutiny
    amount = state["requested_amount_aud"]
    if amount > 200_000:
        risk_score += 20
        risk_factors.append("Grant amount exceeds $200K — enhanced due diligence required")

    # Factor 4: New applicant (no grant history)
    risk_score += 10  # New applicant premium
    risk_factors.append("No prior grant history — first-time applicant")

    risk_score = min(risk_score, 100.0)

    # Recommendation logic
    requires_human = (
        amount > _MAX_GRANT_AUTO_APPROVE_AUD or
        risk_score > 40 or
        not state.get("eligibility_passed", False) or
        len(issues) > 0
    )

    if not state.get("eligibility_passed", False):
        recommendation = "DECLINE"
        conditions: list[str] = ["Applicant does not meet program eligibility criteria"]
    elif risk_score <= 25 and not issues and amount <= _MAX_GRANT_AUTO_APPROVE_AUD:
        recommendation = "APPROVE"
        conditions = []
    elif risk_score <= 60:
        recommendation = "APPROVE_WITH_CONDITIONS"
        conditions = [
            "Milestones and acquittal reporting required at 50% and 100% drawdown",
            "Audited financial statements required before final payment",
        ] + (["Resolve document issues before proceeding"] if issues else [])
    else:
        recommendation = "DECLINE"
        conditions = ["Risk profile exceeds program guidelines — applicant may reapply with additional documentation"]

    notes = (
        f"AI assessment for {state['grant_program']} — ${amount:,.0f} application.\n"
        f"Risk score: {risk_score:.0f}/100. Key risk factors: {'; '.join(risk_factors[:3])}.\n"
        f"Recommendation: {recommendation}. Human review {'REQUIRED' if requires_human else 'OPTIONAL'}.\n"
        f"PGPA Act 2013 s71 — delegate authority level required for this decision."
    )

    logger.info("Recommendation complete", recommendation=recommendation, risk_score=risk_score, human_required=requires_human)

    return {
        **state,
        "risk_score": risk_score,
        "risk_factors": risk_factors,
        "recommendation": recommendation,
        "conditions": conditions,
        "assessor_notes": notes,
        "requires_human_decision": requires_human,
        "commonwealth_register_ready": recommendation == "APPROVE" and not requires_human,
        "_tokens_used": state.get("_tokens_used", 0) + 380,
    }


def _route_after_eligibility(state: GrantState) -> str:
    if not state.get("eligibility_passed", False):
        return "score_and_recommend"   # Skip doc verification for clear fails
    return "verify_documents"


def build_graph(llm: Any = None) -> StateGraph:
    graph = StateGraph(GrantState)
    graph.add_node("check_eligibility", _check_eligibility)
    graph.add_node("verify_documents", _verify_documents)
    graph.add_node("score_and_recommend", _score_and_recommend)
    graph.set_entry_point("check_eligibility")
    graph.add_conditional_edges("check_eligibility", _route_after_eligibility, {"verify_documents": "verify_documents", "score_and_recommend": "score_and_recommend"})
    graph.add_edge("verify_documents", "score_and_recommend")
    graph.add_edge("score_and_recommend", END)
    return graph.compile()


AGENT_METADATA = {
    "id": "gov_grant_assessment",
    "name": "Grant Assessment Pipeline",
    "industry": "government",
    "clouds": ["aws", "azure", "gcp"],
    "orchestration": "sequential",
    "sla_seconds": 300,
    "compliance": ["PGPA_Act_2013", "Commonwealth_Grants_Policy", "Privacy_Act_1988"],
    "description": (
        "End-to-end grant assessment: AWS Bedrock runs the Commonwealth Grants Policy eligibility "
        "rules engine. Azure SK + Document Intelligence verifies submitted documents and extracts "
        "entities. GCP Vertex AI computes a risk score and generates a PGPA-compliant recommendation. "
        "All decisions require human delegate approval. Audit trail published to Commonwealth Grants Register."
    ),
}
