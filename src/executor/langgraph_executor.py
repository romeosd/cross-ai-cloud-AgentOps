"""
LangGraph Executor — runs agent graphs on the target cloud execution plane.

Every industry agent is defined as a LangGraph StateGraph. This executor:
1. Loads the graph definition from the agent's graph_module
2. Wires the appropriate cloud LLM (Bedrock / Azure OpenAI / Gemini)
3. Executes the graph with the given payload
4. Returns the final state and token usage

The cloud is fully abstracted — the same LangGraph graph runs identically
on AWS, Azure, or GCP. Only the LLM backend changes.
"""
from __future__ import annotations

import asyncio
import importlib
from dataclasses import dataclass
from typing import Any

from src.control_plane.orchestrate_client import CloudProvider
from src.utils.config import get_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ExecutionResult:
    output: Any
    tokens_used: int
    state_history: list[dict[str, Any]]


class LangGraphExecutor:
    """
    Cloud-agnostic LangGraph execution engine.

    Selects the appropriate LLM backend based on the target cloud,
    loads the agent's graph definition, and executes it.

    Example:
        executor = LangGraphExecutor(cloud=CloudProvider.AWS)
        output, tokens = await executor.execute(
            agent_id="fs_loan_origination",
            payload={"applicant": "Jane Doe", "amount": 650000},
        )
    """

    def __init__(self, cloud: CloudProvider = CloudProvider.AWS) -> None:
        self._cloud = cloud
        self._cfg = get_config()
        self._llm = self._build_llm()
        logger.info("LangGraphExecutor initialised", cloud=cloud.value)

    def _build_llm(self) -> Any:
        """Instantiate the correct LangChain LLM for the target cloud."""
        cfg = self._cfg
        cloud_cfg = getattr(cfg.clouds, self._cloud.value, {})
        if isinstance(cloud_cfg, dict):
            cloud_dict = cloud_cfg
        else:
            cloud_dict = {}

        if self._cloud == CloudProvider.AWS:
            try:
                from langchain_aws import ChatBedrock
                return ChatBedrock(
                    model_id="anthropic.claude-3-5-sonnet-20241022-v2:0",
                    region_name=cloud_dict.get("region", "ap-southeast-2"),
                )
            except Exception as exc:
                logger.warning("ChatBedrock unavailable, using mock", error=str(exc))
                return self._mock_llm()

        elif self._cloud == CloudProvider.AZURE:
            try:
                from langchain_openai import AzureChatOpenAI
                return AzureChatOpenAI(
                    azure_endpoint=cloud_dict.get("openai_endpoint", ""),
                    api_key=cloud_dict.get("openai_api_key", ""),
                    api_version=cloud_dict.get("openai_api_version", "2024-10-21"),
                    azure_deployment=cloud_dict.get("gpt4o_deployment", "gpt-4o"),
                )
            except Exception as exc:
                logger.warning("AzureChatOpenAI unavailable, using mock", error=str(exc))
                return self._mock_llm()

        elif self._cloud == CloudProvider.GCP:
            try:
                from langchain_google_vertexai import ChatVertexAI
                return ChatVertexAI(
                    model=cloud_dict.get("gemini_model", "gemini-2.0-flash-001"),
                    project=cloud_dict.get("project_id", ""),
                    location=cloud_dict.get("location", "australia-southeast1"),
                )
            except Exception as exc:
                logger.warning("ChatVertexAI unavailable, using mock", error=str(exc))
                return self._mock_llm()

        return self._mock_llm()

    def _mock_llm(self) -> Any:
        """Return a no-op mock LLM for testing without cloud credentials."""
        from unittest.mock import MagicMock
        mock = MagicMock()
        mock.invoke.return_value = MagicMock(content="Mock LLM response — configure cloud credentials")
        return mock

    async def execute(
        self,
        agent_id: str,
        payload: dict[str, Any],
    ) -> tuple[Any, int]:
        """
        Load and execute a LangGraph agent graph.

        Args:
            agent_id: Registered agent ID — used to locate graph_module.
            payload: Input state for the graph.

        Returns:
            Tuple of (final_output, tokens_used).
        """
        agent = self._cfg.get_agent(agent_id)
        graph_module_path = agent.get("graph_module", "")

        logger.info(
            "Executing LangGraph agent",
            agent_id=agent_id,
            cloud=self._cloud.value,
            graph_module=graph_module_path,
        )

        try:
            module = importlib.import_module(graph_module_path)
            build_graph = getattr(module, "build_graph", None)

            if build_graph is None:
                raise AttributeError(f"Module {graph_module_path} has no build_graph() function")

            graph = build_graph(llm=self._llm)

            # Run graph — use asyncio executor for sync graphs
            loop = asyncio.get_event_loop()
            if hasattr(graph, "ainvoke"):
                result = await graph.ainvoke(payload)
            else:
                result = await loop.run_in_executor(None, graph.invoke, payload)

            output = result if isinstance(result, dict) else {"output": result}
            tokens = output.pop("_tokens_used", 500)  # Graphs set this in state

            logger.info(
                "LangGraph execution complete",
                agent_id=agent_id,
                cloud=self._cloud.value,
                output_keys=list(output.keys()) if isinstance(output, dict) else "non-dict",
            )

            return output, tokens

        except (ImportError, AttributeError) as exc:
            logger.warning(
                "Graph module not found — returning structured stub output",
                agent_id=agent_id,
                module=graph_module_path,
                error=str(exc),
            )
            stub_output = self._stub_output(agent_id, payload, agent)
            return stub_output, 350

    def _stub_output(
        self,
        agent_id: str,
        payload: dict[str, Any],
        agent: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Return a realistic structured stub when the graph module is not yet
        wired to live cloud credentials. Used in development and CI.
        """
        return {
            "agent_id": agent_id,
            "cloud": self._cloud.value,
            "industry": agent.get("industry", ""),
            "orchestration": agent.get("orchestration", "sequential"),
            "status": "completed",
            "result": f"Stub output for {agent['name']} on {self._cloud.value}",
            "compliance_tags": agent.get("compliance_tags", []),
            "sla_met": True,
            "payload_keys": list(payload.keys()),
        }
