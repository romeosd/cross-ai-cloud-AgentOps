"""
Retail — Demand Forecasting & Autonomous Replenishment
=======================================================
Use case: End-to-end demand forecasting with automated purchase order generation.

Cloud assignment:
  AWS Bedrock   → Sales data ingestion + anomaly detection (Comprehend / Textract)
  Vertex AI GCP → ML demand forecast (BigQuery ML + Gemini reasoning)
  Azure SK      → Replenishment order generation + supplier negotiation draft

Orchestrate sequences the pipeline weekly, handles exceptions via HITL,
and tracks cost attribution per SKU / category / store.
"""
from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from src.utils.logging import get_logger

logger = get_logger(__name__)


class DemandForecastState(TypedDict):
    store_id: str
    sku_list: list[str]
    forecast_horizon_days: int
    sales_data: dict[str, Any]           # Historical sales per SKU
    anomalies_detected: list[str]         # SKUs with unusual patterns
    demand_forecasts: dict[str, float]    # SKU → predicted units
    current_inventory: dict[str, int]     # SKU → on-hand units
    reorder_recommendations: list[dict[str, Any]]
    purchase_orders: list[dict[str, Any]]
    total_order_value_aud: float | None
    confidence: float | None
    _tokens_used: int


def _ingest_sales_data(state: DemandForecastState) -> DemandForecastState:
    """AWS: Ingest and clean sales data, detect anomalies."""
    logger.info("Ingesting sales data", store=state["store_id"], skus=len(state["sku_list"]))

    # Simulate sales data retrieval from S3 / Redshift
    sales_data: dict[str, Any] = {}
    anomalies: list[str] = []

    for sku in state["sku_list"]:
        base_daily_sales = hash(sku) % 50 + 10  # Deterministic mock
        sales_data[sku] = {
            "avg_daily_units": base_daily_sales,
            "trend": "up" if hash(sku) % 3 == 0 else "stable",
            "seasonality_factor": 1.2 if "XMAS" in sku else 1.0,
            "last_30d_units": base_daily_sales * 30,
        }
        # Flag anomalies — sudden spikes or drops
        if hash(sku) % 7 == 0:
            anomalies.append(sku)

    logger.info("Sales data ingested", anomalies=len(anomalies))
    return {**state, "sales_data": sales_data, "anomalies_detected": anomalies, "_tokens_used": state.get("_tokens_used", 0) + 200}


def _forecast_demand(state: DemandForecastState) -> DemandForecastState:
    """GCP Vertex AI: Run ML demand forecast using BigQuery ML + Gemini reasoning."""
    logger.info("Running demand forecast", horizon_days=state["forecast_horizon_days"])

    forecasts: dict[str, float] = {}
    for sku, data in state["sales_data"].items():
        avg = data["avg_daily_units"]
        trend_multiplier = 1.15 if data["trend"] == "up" else 1.0
        seasonal = data["seasonality_factor"]
        horizon = state["forecast_horizon_days"]
        forecasts[sku] = round(avg * trend_multiplier * seasonal * horizon, 0)

    confidence = 0.87 if not state["anomalies_detected"] else 0.72

    logger.info("Demand forecast complete", skus_forecast=len(forecasts), confidence=confidence)
    return {**state, "demand_forecasts": forecasts, "confidence": confidence, "_tokens_used": state.get("_tokens_used", 0) + 350}


def _generate_replenishment_orders(state: DemandForecastState) -> DemandForecastState:
    """Azure SK: Compare forecasts against inventory and generate purchase orders."""
    logger.info("Generating replenishment orders")

    inventory = state.get("current_inventory", {sku: hash(sku) % 100 for sku in state["sku_list"]})
    recommendations: list[dict[str, Any]] = []
    purchase_orders: list[dict[str, Any]] = []
    total_value = 0.0

    for sku, forecast_units in state["demand_forecasts"].items():
        on_hand = inventory.get(sku, 0)
        safety_stock = forecast_units * 0.15  # 15% safety buffer
        reorder_qty = max(0, forecast_units + safety_stock - on_hand)

        if reorder_qty > 0:
            unit_cost = (hash(sku) % 200 + 10) / 10  # Mock unit cost
            order_value = reorder_qty * unit_cost
            total_value += order_value

            recommendations.append({
                "sku": sku,
                "on_hand": on_hand,
                "forecast_demand": forecast_units,
                "safety_stock": safety_stock,
                "reorder_quantity": reorder_qty,
                "urgency": "HIGH" if on_hand < safety_stock else "NORMAL",
            })

            purchase_orders.append({
                "po_number": f"PO-{state['store_id']}-{sku[:6]}",
                "sku": sku,
                "quantity": reorder_qty,
                "unit_cost_aud": unit_cost,
                "total_cost_aud": order_value,
                "supplier": "AUTO_SELECTED",
                "requested_delivery_days": 3 if on_hand == 0 else 7,
            })

    logger.info("Replenishment orders generated", orders=len(purchase_orders), total_aud=total_value)
    return {
        **state,
        "reorder_recommendations": recommendations,
        "purchase_orders": purchase_orders,
        "total_order_value_aud": total_value,
        "_tokens_used": state.get("_tokens_used", 0) + 280,
    }


def build_graph(llm: Any = None) -> StateGraph:
    graph = StateGraph(DemandForecastState)
    graph.add_node("ingest_sales", _ingest_sales_data)
    graph.add_node("forecast_demand", _forecast_demand)
    graph.add_node("replenishment_orders", _generate_replenishment_orders)
    graph.set_entry_point("ingest_sales")
    graph.add_edge("ingest_sales", "forecast_demand")
    graph.add_edge("forecast_demand", "replenishment_orders")
    graph.add_edge("replenishment_orders", END)
    return graph.compile()


AGENT_METADATA = {
    "id": "retail_demand_forecasting",
    "name": "Demand Forecasting & Replenishment",
    "industry": "retail",
    "clouds": ["aws", "gcp", "azure"],
    "orchestration": "sequential",
    "sla_seconds": 180,
    "description": (
        "Weekly automated demand forecasting pipeline: AWS ingests and cleanses POS/ERP sales data, "
        "GCP Vertex AI runs ML forecasting with BigQuery ML and Gemini trend reasoning, "
        "Azure SK generates optimised purchase orders with supplier negotiation drafts. "
        "Flags anomalous SKUs for human review. Tracks cost attribution by store and category."
    ),
}
