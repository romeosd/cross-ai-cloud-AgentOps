# Cross-Cloud AgentOps — Enterprise AI Agent Control Plane

![Cross-Cloud AgentOps Architecture](docs/images/ibm_multicloud_agentops.png)

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![watsonx](https://img.shields.io/badge/watsonx-Orchestrate-be95ff?logo=ibm)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-green)
![AWS](https://img.shields.io/badge/AWS-Bedrock-orange?logo=amazonaws)
![Azure](https://img.shields.io/badge/Azure-OpenAI%20%7C%20SK-0078d4?logo=microsoftazure)
![GCP](https://img.shields.io/badge/GCP-Vertex%20AI-4285f4?logo=googlecloud)
![License](https://img.shields.io/badge/License-Apache%202.0-green)
![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)

**Author:** Ramy Amer  
**Region:** ap-southeast-2 / australiaeast / australia-southeast1

---

Every enterprise running AI at scale faces the same hidden problem: **agents are proliferating faster than governance can keep up.** An AWS Bedrock Agent doing loan origination. An Azure Semantic Kernel agent handling customer complaints. A Vertex AI Agent Builder answering policy questions. A LangGraph agent detecting fraud. Nobody can answer:

> *What are all my agents doing right now? Which ones are failing? Which decisions did they make? Who authorised them? What did they cost? Are we compliant?*

This repository is the answer. **watsonx Orchestrate acts as the enterprise control plane** — a single pane of glass governing every AI agent across AWS, Azure, and GCP. LangGraph provides the stateful execution engine. Each cloud is an interchangeable execution plane.

---

## The Architecture Story

```mermaid
graph TB
    subgraph ControlPlane["watsonx Orchestrate — Enterprise Control Plane"]
        REG[Agent Registry\n19 registered agents\n5 industries · 3 clouds]
        ROUTE[Intent Router\nCost · Latency · Capability]
        GOV[Governance Engine\nAPRA · ASIC · WHS · Privacy Act]
        AUDIT[Immutable Audit Log\n7-year retention · ANAO compliant]
        DASH[AgentOps Dashboard\nSingle Pane of Glass]
        COST[Cost Tracker\nChargeback by BU · Cloud · Industry]
    end

    subgraph LangGraph["LangGraph Execution Engine"]
        EXEC[Cloud-agnostic\nStateGraph executor]
        STATE[Stateful workflows\nCONDITIONAL routing]
        HITL[Human-in-Loop\nEscalation gates]
    end

    subgraph AWS["AWS Execution Plane"]
        BEDROCK[Bedrock Agents\nClaude Sonnet]
        COMPREHEND[Comprehend NLP]
        TEXTRACT[Textract OCR]
        SM[SageMaker ML]
    end

    subgraph Azure["Azure Execution Plane"]
        AOI[Azure OpenAI\nGPT-4o]
        SK[Semantic Kernel\nAgents + Planners]
        DI[Document Intelligence]
        AML_AZ[Azure ML]
    end

    subgraph GCP["GCP Execution Plane"]
        GEMINI[Vertex AI Gemini\n2.0 Flash / Pro]
        AGENT_B[Agent Builder\nRAG Engine]
        BQML[BigQuery ML]
        DOCAI[Document AI]
    end

    subgraph Industries["Industry Use Cases"]
        FS[Financial Services\nLoan · AML · APRA · Wealth]
        RETAIL[Retail\nDemand · Personalisation · Fraud]
        TELCO[Telco\nFault · Churn · Fraud · Spectrum]
        MINING[Mining\nMaintenance · Safety · Ore · Supply]
        GOV_IND[Government\nCitizen · Grants · Procurement · Benefits]
    end

    Industries --> ControlPlane
    ControlPlane --> LangGraph
    LangGraph --> AWS
    LangGraph --> Azure
    LangGraph --> GCP
    GOV --> AUDIT
    ROUTE --> COST
    DASH --> REG
```

---

## How It Works — The Three Layers

```mermaid
flowchart LR
    subgraph L1["Layer 1 — Intent"]
        USER[Business User\nor System Event]
        INTENT["Natural language intent:\n'Assess a $650K home loan'\n'Network fault on node AUS-01'\n'Suspicious $9,500 cash transaction'"]
    end

    subgraph L2["Layer 2 — Orchestrate Control Plane"]
        MATCH[Intent Matcher\nSemantic trigger matching]
        REGISTRY[Agent Registry\nLookup registered agent]
        ROUTE2[Cloud Router\nCost · Latency · Capability]
        GOVERN[Governance Gate\nPre-execution policy check]
    end

    subgraph L3["Layer 3 — LangGraph Execution"]
        GRAPH[StateGraph\nMulti-step stateful workflow]
        NODES[Specialised Nodes\nAWS → Azure → GCP]
        COND[Conditional Routing\nBranch on intermediate results]
        OUTPUT[Structured Output\n+ Token count + Cost]
    end

    subgraph L4["Layer 4 — Post-execution Governance"]
        GOV2[Governance Engine\nConfidence · PII · Compliance]
        HITL2[Human-in-Loop\nEscalation queue]
        AUDIT2[Audit Trail\nImmutable · 7-year retention]
        METRIC[AgentOps Metrics\nCost · Latency · Success rate]
    end

    USER --> INTENT --> MATCH --> REGISTRY --> ROUTE2 --> GOVERN
    GOVERN --> GRAPH --> NODES --> COND --> OUTPUT
    OUTPUT --> GOV2 --> HITL2
    OUTPUT --> AUDIT2
    OUTPUT --> METRIC
```

---

## Industry Use Cases

### Financial Services — 4 Agents

```mermaid
sequenceDiagram
    participant O as Orchestrate
    participant AWS as AWS Bedrock
    participant AZ as Azure SK
    participant GCP as Vertex AI Gemini
    participant H as Human Officer

    Note over O,H: Loan Origination Pipeline

    O->>AWS: Retrieve credit profile + compute DTI ratio
    AWS-->>O: Credit score: 720 · DTI: 0.32 · Risk tier: AA

    O->>AZ: NCCP responsible lending compliance check
    AZ-->>O: Policy compliant ✓ · No stress-test breach

    O->>GCP: Generate formal decision letter
    GCP-->>O: APPROVE letter drafted · APRA CPS 220 audit stamp

    O->>H: Route to delegate for sign-off (NCCP mandatory)
    H-->>O: Approved ✓

    Note over O,H: AML Detection — Stateful Graph

    O->>AWS: Screen $9,500 CASH transaction (IR counterparty)
    AWS-->>O: Rules fired: STRUCTURING + HIGH_RISK_JURISDICTION
    AWS->>AWS: Enrich → Score (risk: 75/100)
    AWS-->>O: SAR generated · Case AML-TXN001-X7F2
    O->>H: Escalate to compliance officer (AUSTRAC SMR required)
```

### Telco — Network Fault Remediation

```mermaid
stateDiagram-v2
    [*] --> DETECT: Alert received\nNetFlow/SNMP/Synthetic

    DETECT --> TRIAGE: AWS Bedrock\nClassify severity P1-P4

    TRIAGE --> RCA: GCP Vertex AI Gemini\nRoot cause analysis\nover network topology

    RCA --> DECISION: Confidence check

    DECISION --> AUTO_REMEDIATE: Confidence ≥ 75%\nAWS executes network API calls
    DECISION --> ESCALATE_NOC: Confidence < 75%\nRoute to NOC L3

    AUTO_REMEDIATE --> VERIFY: Synthetic tests\nping · traceroute · transactions

    VERIFY --> CLOSED: Tests pass ✓\nMTTR logged · SLA preserved
    VERIFY --> ESCALATE_NOC: Tests fail\nManual intervention

    ESCALATE_NOC --> [*]
    CLOSED --> [*]
```

### Mining — Predictive Maintenance

```mermaid
flowchart TD
    IOT[IoT Sensors\n45 Haul Trucks · 8 Excavators\n3 Crushers · Conveyor systems]

    IOT --> AWS_IOT[AWS IoT Core\nSensor stream ingestion]
    AWS_IOT --> ANOMALY[Anomaly Detection\nThreshold breach classification]

    ANOMALY --> GCP_ML[GCP Vertex AI\nTime-series failure\nprediction model]

    GCP_ML --> PROB{Failure\nprobability?}

    PROB -->|≥ 80%| CRITICAL[CRITICAL\nSchedule within 8hrs\nWHS permit required]
    PROB -->|60-80%| HIGH[HIGH\nSchedule within 24hrs]
    PROB -->|35-60%| MEDIUM[MEDIUM\nSchedule within 72hrs]
    PROB -->|< 35%| LOW[LOW\nContinue monitoring]

    CRITICAL --> WO[Azure SK\nSAP PM Work Order\n+ Parts Requisition]
    HIGH --> WO
    MEDIUM --> WO

    WO --> COST_AVOID["Cost avoidance:\n~$180K AUD/hr\nunplanned downtime avoided"]
```

### Government — Grant Assessment

```mermaid
flowchart LR
    APP([Grant Application\nABN · Documents · Project plan]) --> ELIG

    subgraph AWS["AWS Bedrock — Eligibility Engine"]
        ELIG[Commonwealth Grants\nPolicy rules engine\nABN · Entity type · Amount limits]
    end

    subgraph AZURE["Azure SK + Document Intelligence"]
        VERIFY[Document Verification\nFinancial statements\nEntity resolution · ABR lookup]
    end

    subgraph GCP2["GCP Vertex AI — Risk & Recommendation"]
        SCORE[Risk Scoring\n0-100 composite score]
        REC[Recommendation\nAPPROVE / CONDITIONAL / DECLINE]
    end

    ELIG -->|Pass| VERIFY
    ELIG -->|Fail| SCORE
    VERIFY --> SCORE
    SCORE --> REC

    REC --> HUMAN[Human Delegate\nMandatory — PGPA Act s71]
    HUMAN --> REGISTER[Commonwealth\nGrants Register]

    style HUMAN fill:#ff9900
```

---

## AgentOps — Single Pane of Glass

```mermaid
graph TB
    subgraph AGENTS["All Registered Agents (19 across 5 industries)"]
        A1[fs_loan_origination\nAWS·Azure·GCP]
        A2[fs_aml_detection\nAWS]
        A3[retail_demand_forecasting\nAWS·GCP·Azure]
        A4[telco_network_fault\nAWS·GCP]
        A5[mining_predictive_maintenance\nAWS·GCP·Azure]
        A6[gov_grant_assessment\nAWS·Azure·GCP]
        AN[... 13 more agents ...]
    end

    subgraph DASHBOARD["AgentOps Dashboard — /dashboard/summary"]
        HEALTH[Agent Health\nActive · Degraded · Offline]
        COST2[Cost by Cloud\nAWS · Azure · GCP]
        LATENCY[Latency P95\nper agent · per cloud]
        SUCCESS[Success Rate\n24h rolling window]
        ESCALATIONS2[Escalation Rate\nHuman-in-loop triggers]
        ALERTS2[Active Alerts\nCost threshold · SLA breach]
    end

    subgraph PROMETHEUS["Prometheus + Grafana"]
        PROM[agentops_invocations_total\nagentops_cost_usd_total\nagentops_latency_seconds\nagentops_escalations_total\nagentops_cloud_health]
    end

    AGENTS --> DASHBOARD
    DASHBOARD --> PROMETHEUS
```

---

## Project Structure

```
cross-ai-cloud-AgentOps/
├── config/
│   └── orchestrate_config.yaml          # ← All 19 agents registered here
│
├── src/
│   ├── control_plane/
│   │   ├── orchestrate_client.py        # watsonx Orchestrate integration, routing, workflows
│   │   ├── agent_registry.py            # Agent discovery, health, skill cards
│   │   ├── governance.py                # Policy enforcement, PII, HITL, audit
│   │   └── skill_catalog.py             # Skill definitions for Orchestrate
│   │
│   ├── executor/
│   │   └── langgraph_executor.py        # Cloud-agnostic LangGraph runner
│   │
│   ├── industries/
│   │   ├── financial_services/
│   │   │   ├── loan_origination.py      # ★ Full: AWS credit → Azure policy → GCP letter
│   │   │   ├── aml_detection.py         # ★ Full: Stateful AML graph + SAR generation
│   │   │   ├── regulatory_reporting.py  # APRA/ASIC multi-agent pipeline
│   │   │   └── wealth_advisory.py       # Portfolio analysis agent chain
│   │   │
│   │   ├── retail/
│   │   │   ├── demand_forecasting.py    # ★ Full: AWS ingest → GCP ML → Azure orders
│   │   │   ├── customer_personalisation.py
│   │   │   ├── returns_fraud.py
│   │   │   └── supplier_negotiation.py
│   │   │
│   │   ├── telco/
│   │   │   ├── network_fault_remediation.py  # ★ Full: Triage → RCA → Auto-remediate
│   │   │   ├── churn_intervention.py
│   │   │   └── fraud_detection.py
│   │   │
│   │   ├── mining/
│   │   │   ├── predictive_maintenance.py     # ★ Full: Sensors → Prediction → Work order
│   │   │   ├── safety_compliance.py
│   │   │   ├── ore_grade_optimisation.py
│   │   │   └── supply_chain_disruption.py
│   │   │
│   │   └── government/
│   │       ├── grant_assessment.py           # ★ Full: Eligibility → Docs → Risk → Delegate
│   │       ├── citizen_services.py
│   │       ├── procurement_compliance.py
│   │       └── fraud_benefits.py
│   │
│   └── agentops/
│       ├── dashboard.py                 # FastAPI + Prometheus single pane of glass
│       ├── cost_tracker.py              # Per-agent/cloud/BU cost attribution
│       ├── health_monitor.py            # Agent health polling
│       └── audit_logger.py              # Immutable audit trail
│
├── tests/unit/
│   └── test_agentops.py                 # Registry, governance, cost, graph tests
│
├── notebooks/
│   ├── 01_orchestrate_control_plane.ipynb
│   ├── 02_financial_services_agents.ipynb
│   ├── 03_retail_agents.ipynb
│   ├── 04_telco_agents.ipynb
│   ├── 05_mining_agents.ipynb
│   └── 06_government_agents.ipynb
│
├── pyproject.toml
└── README.md
```

---

## Quick Start

```bash
git clone https://github.com/romeosd/cross-ai-cloud-AgentOps.git
cd cross-ai-cloud-AgentOps

python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

### Configure credentials

```bash
# watsonx Orchestrate
export ORCHESTRATE_API_URL=https://api.us-south.watsonx.ai/v1
export ORCHESTRATE_INSTANCE_ID=your-instance-id
export ORCHESTRATE_API_KEY=your-api-key

# AWS
export AWS_DEFAULT_REGION=ap-southeast-2
export AWS_BEDROCK_AGENT_ID=your-agent-id

# Azure
export AZURE_OPENAI_ENDPOINT=https://your-name.openai.azure.com/
export AZURE_OPENAI_API_KEY=your-key

# GCP
export GCP_PROJECT_ID=your-project-id
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/sa-key.json
```

### Run a financial services workflow

```python
from src.control_plane.orchestrate_client import OrchestrateClient
import asyncio

client = OrchestrateClient()

result = asyncio.run(client.invoke_workflow(
    agent_id="fs_loan_origination",
    payload={
        "applicant_name": "Jane Doe",
        "loan_amount": 650000,
        "loan_term_years": 30,
        "annual_income": 145000,
        "existing_debts": 12000,
    }
))

print(f"Decision: {result.final_output['decision']}")
print(f"Clouds used: {result.clouds_used}")
print(f"Total cost: ${result.total_cost_usd:.4f}")
print(f"Human approval required: {result.human_approvals_required > 0}")
```

### Run via intent routing

```python
result = asyncio.run(client.route_by_intent(
    "I need to assess a $650,000 home loan application for Jane Doe",
    context={"branch": "Melbourne CBD", "channel": "broker"},
))
```

### Start the AgentOps Dashboard

```bash
uvicorn src.agentops.dashboard:app --host 0.0.0.0 --port 8080

# Endpoints:
# GET /dashboard/summary          — full metrics snapshot
# GET /dashboard/agents/{id}      — single agent detail
# GET /dashboard/registry         — skill catalog
# GET /metrics/prometheus         — Prometheus scrape endpoint
```

### Run tests

```bash
pytest tests/unit/ -v
```

---

## Registered Agents — Full Catalog

| Industry | Agent ID | Clouds | Orchestration | SLA | Human Approval |
|----------|----------|--------|---------------|-----|----------------|
| **Financial Services** | `fs_loan_origination` | AWS·Azure·GCP | Sequential | 120s | ✓ NCCP |
| | `fs_aml_detection` | AWS | Stateful graph | 30s | ✓ AUSTRAC |
| | `fs_regulatory_reporting` | AWS·Azure·GCP | Parallel→Merge | 600s | ✓ APRA |
| | `fs_wealth_advisory` | GCP·Azure | Sequential | 60s | — |
| **Retail** | `retail_demand_forecasting` | AWS·GCP·Azure | Sequential | 180s | — |
| | `retail_personalisation` | GCP·AWS | Stateful graph | 5s | — |
| | `retail_returns_fraud` | AWS·Azure | Sequential | 10s | ✓ |
| | `retail_supplier_negotiation` | GCP·Azure | Stateful graph | 300s | — |
| **Telco** | `telco_network_fault` | AWS·GCP | Stateful graph | 60s | — |
| | `telco_churn_intervention` | GCP·Azure·AWS | Sequential | 15s | — |
| | `telco_fraud_detection` | AWS·Azure | Parallel | 5s | ✓ |
| **Mining** | `mining_predictive_maintenance` | AWS·GCP·Azure | Stateful graph | 30s | WHS CRITICAL |
| | `mining_safety_compliance` | GCP·AWS | Sequential | 10s | ✓ WHS |
| | `mining_ore_optimisation` | GCP·AWS | Sequential | 120s | — |
| | `mining_supply_chain` | Azure·GCP | Stateful graph | 180s | — |
| **Government** | `gov_citizen_services` | Azure·AWS | Sequential | 10s | — |
| | `gov_grant_assessment` | AWS·Azure·GCP | Sequential | 300s | ✓ PGPA |
| | `gov_procurement_compliance` | Azure·GCP | Sequential | 240s | ✓ CPRs |
| | `gov_benefits_fraud` | AWS·Azure | Parallel→Merge | 60s | ✓ Privacy Act |

---

## Governance & Compliance

| Standard | Coverage |
|----------|----------|
| APRA CPS 220 | Risk management — risk scoring, escalation thresholds, decision audit |
| APRA CPS 230 | Operational resilience — circuit breakers, multi-cloud fallback |
| APRA CPS 234 | Information security — PII controls, 7-year audit retention |
| AUSTRAC AML/CTF Act | AML rules engine, SAR generation, FATF typology detection |
| NCCP Act 2009 | Responsible lending — DTI limits, stress-test buffer (rate + 3%) |
| PGPA Act 2013 | Grant delegate authority — mandatory human sign-off |
| WHS Act 2011 | Mining safety — CRITICAL work order escalation, permit requirements |
| Privacy Act 1988 | PII detection and redaction in all agent outputs |
| ISO/IEC 42001 | AI governance documentation and audit trail |

---

## Cost Optimisation by Cloud

| Cloud | Model | Input/1K tokens | Output/1K tokens | Best for |
|-------|-------|----------------|-----------------|---------|
| AWS | Claude 3.5 Sonnet | $0.003 | $0.015 | Complex reasoning, long documents |
| Azure | GPT-4o | $0.0025 | $0.010 | Structured output, function calling |
| GCP | Gemini 2.0 Flash | $0.000075 | $0.000300 | High-volume classification, extraction |

Orchestrate's cost-optimised routing automatically routes high-volume, simple tasks to GCP (50× cheaper per token than AWS for these workloads) and reserves AWS/Azure for tasks requiring complex reasoning or document analysis.

---

*Built by Ramy Amer — Cross-Cloud AgentOps | watsonx Orchestrate + LangGraph + AWS + Azure + GCP*
