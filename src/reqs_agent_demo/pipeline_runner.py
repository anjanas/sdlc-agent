"""Shared wiring + invoke helper for CLI and mock HTTP demos."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver

from reqs_agent_demo.agent.graph import GraphDeps, compile_demo_graph, initialise_state
from reqs_agent_demo.connectors.factories import create_demo_clients
from reqs_agent_demo.paths import config_path, fixtures_path


def build_graph_deps_for_run(
    *,
    page_id: str,
    fixture_mode: bool,
    mock_origin: str,
    max_repairs: int,
    model: str,
) -> GraphDeps:
    if fixture_mode:
        _, _, knowledge_connector, jira_connector = create_demo_clients(None, mock_origin, None, None)

        return GraphDeps(
            confluence=None,
            policy_http=None,
            knowledge=knowledge_connector,
            jira=jira_connector,
            jira_field_map=json.loads(config_path("jira-field-map.json").read_text()),
            page_id=page_id,
            offline_policy_path=fixtures_path("policy", "demo.json"),
            offline_confluence_fixture=fixtures_path("confluence", "demo-prd.json"),
            openai_model=model,
            max_repairs=max_repairs,
            approver_id=os.getenv("APPROVER_ID"),
            approver_email=os.getenv("APPROVER_EMAIL"),
        )

    conf, pol_http, knowledge_connector, jira_connector = create_demo_clients(
        mock_origin,
        mock_origin,
        mock_origin,
        mock_origin,
        token_policy=os.getenv("POLICY_SERVICE_TOKEN"),
        token_knowledge=os.getenv("KNOWLEDGE_SERVICE_TOKEN"),
        tenant_policy=os.getenv("POLICY_TENANT"),
        token_jira=os.getenv("JIRA_MOCK_TOKEN"),
    )

    return GraphDeps(
        confluence=conf,
        policy_http=pol_http,
        knowledge=knowledge_connector,
        jira=jira_connector,
        jira_field_map=json.loads(config_path("jira-field-map.json").read_text()),
        page_id=page_id,
        offline_policy_path=None,
        offline_confluence_fixture=None,
        openai_model=model,
        max_repairs=max_repairs,
        approver_id=os.getenv("APPROVER_ID"),
        approver_email=os.getenv("APPROVER_EMAIL"),
    )


def close_graph_clients(deps: GraphDeps) -> None:
    deps.knowledge.close()
    deps.jira.close()
    if deps.confluence:
        deps.confluence.close()
    if deps.policy_http:
        deps.policy_http.close()


def run_generation_invoke(
    *,
    page_id: str,
    offline_llm: bool,
    approve_path_txt: str | None,
    auto_approve: bool,
    fixture_mode: bool,
    mock_origin: str,
    max_repairs: int,
    model: str,
    checkpointer: BaseCheckpointSaver | None = None,
) -> tuple[dict[str, Any], GraphDeps, Any, Any]:
    """First invoke (+ graph + runnable_config for resume Command)."""

    deps = build_graph_deps_for_run(
        page_id=page_id,
        fixture_mode=fixture_mode,
        mock_origin=mock_origin.rstrip("/"),
        max_repairs=max_repairs,
        model=model,
    )
    runner_graph = compile_demo_graph(deps, checkpointer=checkpointer)

    initial_board = initialise_state(
        page_id=page_id,
        offline_llm=offline_llm,
        approve_with_path=approve_path_txt,
        auto_approve=auto_approve,
    )
    runnable_config = {"configurable": {"thread_id": initial_board["run_id"]}}

    outcome_bundle = runner_graph.invoke(initial_board, runnable_config)
    return outcome_bundle, deps, runner_graph, runnable_config
