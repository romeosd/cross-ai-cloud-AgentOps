"""Unit tests for AgentRegistry, GovernanceEngine, CostTracker, and industry graphs."""
from __future__ import annotations
from unittest.mock import patch, MagicMock
import pytest

# ── AgentRegistry ────────────────────────────────────────────────────────────

def _mock_cfg():
    cfg = MagicMock()
    cfg.orchestrate.governance = {"human_in_loop_threshold": 0.70, "audit_all_invocations": True, "pii_detection_enabled": True, "max_agent_hops": 10}
    cfg.agentops = {"cost_alert_threshold_usd": 100.0, "retention_days": 2555}
    cfg.agent_registry = [
        {"id": "fs_loan_origination", "name": "Loan Origination", "industry": "financial_services",
         "clouds": ["aws","azure","gcp"], "orchestration": "sequential", "graph_module": "src.industries.financial_services.loan_origination",
         "trigger_intent": ["assess loan","loan application"], "sla_seconds": 120, "requires_human_approval": True, "compliance_tags": ["APRA","NCCP"]},
        {"id": "retail_demand_forecasting", "name": "Demand Forecasting", "industry": "retail",
         "clouds": ["aws","gcp","azure"], "orchestration": "sequential", "graph_module": "src.industries.retail.demand_forecasting",
         "trigger_intent": ["forecast demand","replenish stock"], "sla_seconds": 180, "requires_human_approval": False, "compliance_tags": []},
        {"id": "telco_network_fault", "name": "Network Fault", "industry": "telco",
         "clouds": ["aws","gcp"], "orchestration": "stateful_graph", "graph_module": "src.industries.telco.network_fault_remediation",
         "trigger_intent": ["network fault","outage"], "sla_seconds": 60, "requires_human_approval": False, "compliance_tags": []},
    ]
    cfg.get_agent = lambda aid: next(a for a in cfg.agent_registry if a["id"] == aid)
    cfg.agents_by_industry = lambda ind: [a for a in cfg.agent_registry if a.get("industry") == ind]
    return cfg


class TestAgentRegistry:
    @patch("src.control_plane.agent_registry.get_config")
    def test_list_by_industry(self, mock_get):
        from src.control_plane.agent_registry import AgentRegistry
        mock_get.return_value = _mock_cfg()
        registry = AgentRegistry()
        fs_agents = registry.list_by_industry("financial_services")
        assert len(fs_agents) == 1
        assert fs_agents[0].id == "fs_loan_origination"

    @patch("src.control_plane.agent_registry.get_config")
    def test_list_by_cloud(self, mock_get):
        from src.control_plane.agent_registry import AgentRegistry
        mock_get.return_value = _mock_cfg()
        registry = AgentRegistry()
        gcp_agents = registry.list_by_cloud("gcp")
        assert len(gcp_agents) >= 2  # loan_origination and demand_forecasting and network_fault

    @patch("src.control_plane.agent_registry.get_config")
    def test_get_existing_agent(self, mock_get):
        from src.control_plane.agent_registry import AgentRegistry
        mock_get.return_value = _mock_cfg()
        registry = AgentRegistry()
        agent = registry.get("fs_loan_origination")
        assert agent.name == "Loan Origination"
        assert agent.requires_human_approval is True
        assert "APRA" in agent.compliance_tags

    @patch("src.control_plane.agent_registry.get_config")
    def test_get_missing_agent_raises(self, mock_get):
        from src.control_plane.agent_registry import AgentRegistry
        mock_get.return_value = _mock_cfg()
        registry = AgentRegistry()
        with pytest.raises(KeyError):
            registry.get("nonexistent_agent")

    @patch("src.control_plane.agent_registry.get_config")
    def test_summary_counts(self, mock_get):
        from src.control_plane.agent_registry import AgentRegistry
        mock_get.return_value = _mock_cfg()
        registry = AgentRegistry()
        summary = registry.summary()
        assert summary["total_agents"] == 3
        assert "financial_services" in summary["by_industry"]
        assert summary["requiring_human_approval"] == 1


# ── GovernanceEngine ─────────────────────────────────────────────────────────

class TestGovernanceEngine:
    @patch("src.control_plane.governance.get_config")
    def test_high_confidence_approved(self, mock_get):
        from src.control_plane.governance import GovernanceEngine
        mock_get.return_value = _mock_cfg()
        gov = GovernanceEngine()
        decision = gov.evaluate(
            agent_id="retail_demand_forecasting", industry="retail", cloud="gcp",
            confidence=0.92, output={"orders": 5}, cost_usd=0.01, workflow_id="wf-001",
        )
        assert decision.approved is True
        assert decision.escalate_to_human is False
        assert decision.pii_detected is False

    @patch("src.control_plane.governance.get_config")
    def test_low_confidence_escalates(self, mock_get):
        from src.control_plane.governance import GovernanceEngine
        mock_get.return_value = _mock_cfg()
        gov = GovernanceEngine()
        decision = gov.evaluate(
            agent_id="fs_loan_origination", industry="financial_services", cloud="aws",
            confidence=0.55, output={"decision": "APPROVE"}, cost_usd=0.04, workflow_id="wf-002",
            compliance_tags=["APRA", "NCCP"],
        )
        assert decision.escalate_to_human is True

    @patch("src.control_plane.governance.get_config")
    def test_pii_detected_in_output(self, mock_get):
        from src.control_plane.governance import GovernanceEngine
        mock_get.return_value = _mock_cfg()
        gov = GovernanceEngine()
        decision = gov.evaluate(
            agent_id="fs_aml_detection", industry="financial_services", cloud="aws",
            confidence=0.88, output={"report": "Contact john@example.com TFN: 123456789"},
            cost_usd=0.02, workflow_id="wf-003",
        )
        assert decision.pii_detected is True

    @patch("src.control_plane.governance.get_config")
    def test_risk_score_increases_with_violations(self, mock_get):
        from src.control_plane.governance import GovernanceEngine
        mock_get.return_value = _mock_cfg()
        gov = GovernanceEngine()
        d1 = gov.evaluate("a1", "retail", "gcp", 0.95, {}, 0.01, "wf1")
        d2 = gov.evaluate("a2", "financial_services", "aws", 0.50, {}, 0.01, "wf2", compliance_tags=["APRA","ASIC"])
        assert d2.risk_score > d1.risk_score


# ── CostTracker ───────────────────────────────────────────────────────────────

class TestCostTracker:
    @patch("src.agentops.cost_tracker.get_config")
    def test_record_and_report(self, mock_get):
        from src.agentops.cost_tracker import CostTracker
        mock_get.return_value = _mock_cfg()
        tracker = CostTracker()
        tracker.record("fs_loan_origination", "aws", "financial_services", 800, 600, business_unit="Retail_Banking")
        tracker.record("retail_demand_forecasting", "gcp", "retail", 500, 300, business_unit="Merchandise")
        report = tracker.monthly_report()
        assert report.total_usd > 0
        assert "aws" in report.by_cloud or "gcp" in report.by_cloud
        assert "financial_services" in report.by_industry

    @patch("src.agentops.cost_tracker.get_config")
    def test_cheapest_cloud_is_gcp(self, mock_get):
        from src.agentops.cost_tracker import CostTracker
        mock_get.return_value = _mock_cfg()
        tracker = CostTracker()
        cheapest = tracker.cheapest_cloud_for_tokens(1000, 500)
        assert cheapest == "gcp"

    @patch("src.agentops.cost_tracker.get_config")
    def test_auto_compute_cost(self, mock_get):
        from src.agentops.cost_tracker import CostTracker
        mock_get.return_value = _mock_cfg()
        tracker = CostTracker()
        entry = tracker.record("test_agent", "aws", "retail", tokens_input=1000, tokens_output=500)
        expected = (1000/1000 * 0.003) + (500/1000 * 0.015)
        assert abs(entry.cost_usd - expected) < 0.0001


# ── Loan Origination Graph ────────────────────────────────────────────────────

class TestLoanOriginationGraph:
    def test_build_graph_compiles(self):
        from src.industries.financial_services.loan_origination import build_graph
        graph = build_graph()
        assert graph is not None

    def test_approve_high_credit_score(self):
        from src.industries.financial_services.loan_origination import build_graph
        graph = build_graph()
        result = graph.invoke({
            "applicant_name": "Jane Doe",
            "loan_amount": 400000,
            "loan_term_years": 30,
            "annual_income": 120000,
            "existing_debts": 10000,
            "credit_score": None,
            "dti_ratio": None,
            "policy_compliant": None,
            "risk_tier": None,
            "decision": None,
            "decision_letter": None,
            "conditions": [],
            "compliance_flags": [],
            "_tokens_used": 0,
        })
        assert result["decision"] in ("APPROVE", "CONDITIONAL", "DECLINE")
        assert result["decision_letter"] is not None
        assert result["_tokens_used"] > 0


# ── AML Detection Graph ───────────────────────────────────────────────────────

class TestAMLDetectionGraph:
    def test_high_risk_transaction_generates_sar(self):
        from src.industries.financial_services.aml_detection import build_graph
        graph = build_graph()
        result = graph.invoke({
            "transaction_id": "TXN-001",
            "account_id": "ACC-12345",
            "amount_aud": 9500.0,   # Just below $10K — structuring indicator
            "transaction_type": "CASH",
            "counterparty_country": "IR",  # High-risk jurisdiction
            "alert_triggered": None,
            "alert_rules_fired": [],
            "enriched_data": {},
            "risk_score": None,
            "typologies_matched": [],
            "suspicious_activity_report": None,
            "aml_decision": None,
            "case_id": None,
            "_tokens_used": 0,
        })
        assert result["alert_triggered"] is True
        assert result["risk_score"] is not None
        assert result["risk_score"] > 0

    def test_low_risk_domestic_cleared(self):
        from src.industries.financial_services.aml_detection import build_graph
        graph = build_graph()
        result = graph.invoke({
            "transaction_id": "TXN-002",
            "account_id": "ACC-99999",
            "amount_aud": 500.0,
            "transaction_type": "CARD",
            "counterparty_country": "AU",
            "alert_triggered": None,
            "alert_rules_fired": [],
            "enriched_data": {},
            "risk_score": None,
            "typologies_matched": [],
            "suspicious_activity_report": None,
            "aml_decision": None,
            "case_id": None,
            "_tokens_used": 0,
        })
        assert result["aml_decision"] == "CLEAR"


# ── Mining Predictive Maintenance Graph ───────────────────────────────────────

class TestPredictiveMaintenanceGraph:
    def test_critical_sensor_generates_work_order(self):
        from src.industries.mining.predictive_maintenance import build_graph
        graph = build_graph()
        result = graph.invoke({
            "equipment_id": "HT-045",
            "equipment_type": "HAUL_TRUCK",
            "site_id": "SITE-01",
            "sensor_readings": {
                "vibration_mm_s": 14.0,       # Critical (>12)
                "bearing_temp_celsius": 102.0, # Critical (>100)
                "oil_pressure_bar": 1.1,       # Critical (<1.2)
            },
            "baseline_values": {
                "vibration_mm_s": 4.0,
                "bearing_temp_celsius": 65.0,
                "oil_pressure_bar": 3.5,
            },
            "anomalies": [],
            "failure_probability": None,
            "predicted_failure_hours": None,
            "failure_mode": None,
            "maintenance_priority": None,
            "work_order": None,
            "parts_required": [],
            "estimated_downtime_hours": None,
            "cost_avoidance_aud": None,
            "_tokens_used": 0,
        })
        assert result["maintenance_priority"] in ("CRITICAL", "HIGH", "MEDIUM")
        assert result["work_order"] is not None
        assert result["failure_probability"] > 0
