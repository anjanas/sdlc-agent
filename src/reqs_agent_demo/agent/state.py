"""Typed graph state."""

from __future__ import annotations

from typing import Any

from typing_extensions import TypedDict


class AgentState(TypedDict, total=False):
    run_id: str
    thread_id: str
    auto_approve: bool
    offline_llm: bool
    approve_with_path: str | None
    page_id: str

    policy: dict[str, Any]
    etag_headers: dict[str, str]

    prd_html: str
    prd_title: str
    prd_plain: str
    prd_breadcrumbs: list[dict[str, Any]]
    retrieval_query: str

    assembled_prompt: str

    context_pack: dict[str, Any]
    retrieval_records_path: str

    validator_hint: str
    rubric_soft: str
    repair_hints: list[str]
    llm_attempts: int

    llm_bundle: dict[str, Any]

    validated_stories: list[dict[str, Any]]
    proposal_md_path: str
    proposal_json_path: str
    transcript_path: str
    approvals_ledger_path: str

    approver_meta: dict[str, Any]
    pause_payload: dict[str, Any]
    approval_granted: bool

    jira_projection: dict[str, Any]
    transcript: dict[str, Any]
