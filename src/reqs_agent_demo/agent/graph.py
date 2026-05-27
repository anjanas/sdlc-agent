"""LangGraph orchestration for PRD → governance → RAG → stories → approval → Jira."""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from reqs_agent_demo.agent.heuristic import deterministic_stories_from_prd
from reqs_agent_demo.agent.state import AgentState
from reqs_agent_demo.connectors.confluence import ConfluenceConnector, offline_confluence_fixture
from reqs_agent_demo.connectors.jira import ApprovedBacklog, ApprovedStory, JiraConnector
from reqs_agent_demo.connectors.knowledge import ContextPack, KnowledgeRetriever, RetrievalChunk
from reqs_agent_demo.connectors.policy_client import (
    StoryPolicyClient,
    envelope_to_primitive,
    load_policy_from_file,
    policy_prompt_fragments,
)
from reqs_agent_demo.paths import project_root, prompt_path
from reqs_agent_demo.policy_models import StoryPolicyEnvelope, coerce_stories, stories_bundle_model
from reqs_agent_demo.util.html import heuristic_query


def initialise_state(
    *,
    page_id: str,
    offline_llm: bool,
    approve_with_path: str | None,
    auto_approve: bool,
    run_id: str | None = None,
) -> AgentState:
    rid = run_id or str(uuid.uuid4())
    board: AgentState = {
        "run_id": rid,
        "thread_id": rid,
        "page_id": page_id,
        "offline_llm": offline_llm,
        "approve_with_path": approve_with_path,
        "auto_approve": auto_approve,
        "llm_attempts": 0,
        "repair_hints": [],
    }
    return board


@dataclass(slots=True)
class GraphDeps:
    confluence: ConfluenceConnector | None
    policy_http: StoryPolicyClient | None
    knowledge: KnowledgeRetriever
    jira: JiraConnector
    jira_field_map: dict[str, Any]
    page_id: str
    offline_policy_path: Path | None
    offline_confluence_fixture: Path | None
    openai_model: str
    max_repairs: int
    approver_id: str | None
    approver_email: str | None


def _run_paths(run_id: str) -> tuple[Path, Path, Path]:
    root = project_root()
    rd = root / "runs" / run_id
    approvals = root / "approvals"
    rd.mkdir(parents=True, exist_ok=True)
    approvals.mkdir(parents=True, exist_ok=True)
    return rd, approvals, root


def _restore_context_pack(ctx: dict[str, Any]) -> ContextPack:
    chunks = [
        RetrievalChunk(
            id=c["id"],
            doc_type=c["doc_type"],
            score=float(c.get("score", 0.0)),
            excerpt=c["excerpt"],
            citations=c.get("citations") or [],
        )
        for c in ctx.get("chunks", [])
    ]
    return ContextPack(query=str(ctx.get("query", "")), chunks=chunks)


def _fetch_policy(state: AgentState, deps: GraphDeps) -> dict[str, Any]:
    if deps.policy_http is not None:
        env, hdrs = deps.policy_http.fetch_story_policy()
        return {
            "policy": envelope_to_primitive(env),
            "etag_headers": {k: v for k, v in hdrs.items() if k.lower().startswith(("etag", "cache"))},
        }
    path = deps.offline_policy_path
    if path is None:
        raise RuntimeError("offline_policy_path required when policy_http is unset")
    env = load_policy_from_file(path)
    return {"policy": envelope_to_primitive(env), "etag_headers": {}}


def _ingest_confluence(state: AgentState, deps: GraphDeps) -> dict[str, Any]:
    pid = state.get("page_id") or deps.page_id
    if deps.confluence is not None:
        page = deps.confluence.get_page(pid)
        return {
            "prd_html": page["html"],
            "prd_title": page["title"],
            "prd_plain": page["plain_text"],
            "prd_breadcrumbs": page["breadcrumbs"],
        }
    cpath = deps.offline_confluence_fixture
    if cpath is None:
        raise RuntimeError("offline_confluence_fixture required when confluence is unset")
    page = offline_confluence_fixture(cpath)
    return {
        "prd_html": page["html"],
        "prd_title": page["title"],
        "prd_plain": page["plain_text"],
        "prd_breadcrumbs": page["breadcrumbs"],
    }


def _retrieval_query_node(state: AgentState) -> dict[str, Any]:
    q = heuristic_query(state["prd_title"], state["prd_breadcrumbs"], state["prd_plain"])
    return {"retrieval_query": q}


def _rag_retrieve(state: AgentState, deps: GraphDeps) -> dict[str, Any]:
    top_k_raw = os.getenv("KNOWLEDGE_TOP_K", "8").strip()
    try:
        top_k = int(top_k_raw)
    except ValueError:
        top_k = 8
    pack = deps.knowledge.retrieve(state["retrieval_query"], top_k=top_k)
    rd, _, _ = _run_paths(state["run_id"])
    path_ret = rd / "retrieval.json"
    serial = pack.to_serialisable()
    path_ret.write_text(json.dumps(serial, indent=2), encoding="utf-8")
    return {"context_pack": serial, "retrieval_records_path": str(path_ret)}


def _load_approve_stories(state: AgentState) -> dict[str, Any]:
    path_txt = state.get("approve_with_path")
    if not path_txt:
        raise RuntimeError("approve_with_path missing")
    raw_any = json.loads(Path(path_txt).expanduser().read_text(encoding="utf-8"))
    payload = raw_any if isinstance(raw_any, dict) else {"stories": raw_any}
    validated, errs = coerce_stories(state["policy"], payload)
    if not validated:
        msgs = " | ".join(str(e) for e in errs)
        raise ValueError(f"--approve-with payload failed validation: {msgs}")
    return {"validated_stories": validated, "llm_attempts": state.get("llm_attempts", 0), "repair_hints": []}


def _assemble_prompt(state: AgentState, deps: GraphDeps) -> dict[str, Any]:
    env = StoryPolicyEnvelope.model_validate(state["policy"])
    frags = policy_prompt_fragments(env)
    rubric = prompt_path("rubric.md").read_text(encoding="utf-8")
    pack_snip = _restore_context_pack(state["context_pack"]).to_system_snippet()
    hints = "\n".join(state.get("repair_hints", []) or [])
    hints_block = f"\n### Validator corrections (must fix)\n{hints}\n" if hints.strip() else ""
    prd_excerpt = (state.get("prd_plain") or "")[:6000]

    assembled = (
        "\n".join(frags)
        + "\n\n### Rubric\n"
        + rubric
        + hints_block
        + "\n\n### Retrieved organisational context\n"
        + pack_snip
        + "\n\n### PRD excerpts\n"
        + prd_excerpt
        + "\n\n### Task\n"
        + "Produce user stories compliant with StoryPolicy bounds and enums.\n"
        + "Respond only with structured JSON matching the schema you were given.\n"
    )
    return {"assembled_prompt": assembled, "rubric_soft": rubric[:120]}


def _synthesize_llm(state: AgentState, deps: GraphDeps) -> dict[str, Any]:
    policy = state["policy"]
    html = state.get("prd_html") or ""

    if state.get("offline_llm"):
        bundle_local = deterministic_stories_from_prd(policy, html=html)
    else:
        _, BundleCls = stories_bundle_model(policy)
        llm = ChatOpenAI(model=deps.openai_model, temperature=0.1)
        structured = llm.with_structured_output(BundleCls)
        msgs = [
            SystemMessage(
                content="You draft Jira-ready user stories governed by authoritative JSON policy enums."
            ),
            HumanMessage(content=state["assembled_prompt"]),
        ]
        parsed = structured.invoke(msgs)
        bundle_local = parsed.model_dump()

    # Clear previous validation artefacts for this synthesis round
    return {"llm_bundle": bundle_local}


def _validate_stories(state: AgentState) -> dict[str, Any]:
    policy = state["policy"]
    bundle = state.get("llm_bundle") or {}

    validated, errs = coerce_stories(policy, bundle)

    attempts = int(state.get("llm_attempts", 0))

    if validated:
        return {
            "validated_stories": validated,
            "validator_hint": "",
        }

    errs_text = " | ".join(str(e) for e in errs)

    repaired_hints = list(state.get("repair_hints", []) or [])
    repaired_hints.append(errs_text[:2400])

    return {
        "validated_stories": [],
        "validator_hint": errs_text,
        "repair_hints": repaired_hints,
        "llm_attempts": attempts + 1,
    }


RoutePostValidate = Literal["assemble_prompt", "prepare_proposal", "fail_hard"]


def _route_post_validate(state: AgentState, deps: GraphDeps) -> RoutePostValidate:
    if state.get("validated_stories"):
        return "prepare_proposal"

    tries = int(state.get("llm_attempts", 0))
    if tries > deps.max_repairs:
        return "fail_hard"

    return "assemble_prompt"


def _prepare_proposal(state: AgentState) -> dict[str, Any]:
    run_id = state["run_id"]
    rd, _, _ = _run_paths(run_id)
    stories = state["validated_stories"]

    cp = state.get("context_pack")
    pack_restore = _restore_context_pack(cp) if isinstance(cp, dict) else ContextPack(query="", chunks=[])
    citations = pack_restore.list_citations()

    md_lines = [
        f"# Proposal for run `{run_id}`",
        "",
        "## Stories",
        "",
    ]

    for st in stories:
        md_lines.append(f"### {st.get('summary', '')}")
        md_lines.extend(f"- {b}" for b in st.get("acceptanceCriteria") or [])
        md_lines.append("")

    if citations:
        md_lines.extend(["## Citations used", *[f"- {c}" for c in citations[:120]], ""])

    md_path = rd / "proposal.md"
    json_path = rd / "proposal.json"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    envelope = {"stories": stories, "policy_revision": StoryPolicyEnvelope.model_validate(state["policy"]).revision}
    json_path.write_text(json.dumps(envelope, indent=2), encoding="utf-8")

    return {
        "proposal_md_path": str(md_path),
        "proposal_json_path": str(json_path),
        "pause_payload": {
            "run_id": run_id,
            "story_count": len(stories),
            "proposal_md_path": str(md_path),
            "proposal_json_path": str(json_path),
        },
    }


def _human_gate(state: AgentState) -> dict[str, Any]:
    pause = dict(state.get("pause_payload") or {})

    if state.get("auto_approve"):
        return {"approval_granted": True}

    resume = interrupt(pause)

    approve = False
    story_descriptions: list[str] | None = None

    if isinstance(resume, dict):
        approve = bool(resume.get("approve"))
        raw = resume.get("story_descriptions")
        if isinstance(raw, list):
            story_descriptions = ["" if x is None else str(x) for x in raw]
    elif isinstance(resume, bool):
        approve = resume

    out: dict[str, Any] = {"approval_granted": approve}

    if approve and story_descriptions is not None:
        stories_in = state.get("validated_stories") or []
        if stories_in:
            merged: list[dict[str, Any]] = []
            for i, st in enumerate(stories_in):
                row = dict(st) if isinstance(st, dict) else {}
                if i < len(story_descriptions):
                    trimmed = story_descriptions[i].strip()
                    row["description"] = trimmed if trimmed else None
                merged.append(row)
            out["validated_stories"] = merged

    return out


def _fail_hard(state: AgentState) -> dict[str, Any]:
    rd, _, _ = _run_paths(state["run_id"])
    fail_path = rd / "transcript.failure.json"
    payload = {
        "run_id": state["run_id"],
        "attempts": state.get("llm_attempts", 0),
        "hint": state.get("validator_hint", ""),
    }
    fail_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {"transcript_path": str(fail_path), "jira_projection": {"issues": [], "error": "validation_exhausted"}}


def _ledger_write(state: AgentState, deps: GraphDeps) -> dict[str, Any]:
    _, approvals_dir, _ = _run_paths(state["run_id"])

    backlog = ApprovedBacklog(
        run_id=state["run_id"],
        approved_by_id=deps.approver_id,
        approved_by_email=deps.approver_email,
        stories=[ApprovedStory.model_validate(s) for s in state["validated_stories"]],
    )
    apath = approvals_dir / f"{state['run_id']}.json"
    apath.write_text(backlog.model_dump_json(indent=2), encoding="utf-8")
    return {"approvals_ledger_path": str(apath), "approver_meta": backlog.model_dump()}


def _jira_sink(state: AgentState, deps: GraphDeps) -> dict[str, Any]:
    backlog = ApprovedBacklog(
        run_id=state["run_id"],
        approved_by_id=deps.approver_id,
        approved_by_email=deps.approver_email,
        stories=[ApprovedStory.model_validate(s) for s in state["validated_stories"]],
    )
    results = deps.jira.create_stories(backlog, deps.jira_field_map)

    rd, _, _ = _run_paths(state["run_id"])
    transcript = {
        "run_id": state["run_id"],
        "jira_issues": results,
        "proposal_md_path": state.get("proposal_md_path"),
        "approvals_ledger_path": state.get("approvals_ledger_path"),
        "retrieval_path": state.get("retrieval_records_path"),
        "validated_story_count": len(state.get("validated_stories") or []),
    }
    tpath = rd / "transcript.json"
    tpath.write_text(json.dumps(transcript, indent=2), encoding="utf-8")
    return {"transcript_path": str(tpath), "jira_projection": transcript}


def _reject_finalize(state: AgentState) -> dict[str, Any]:
    rd, _, _ = _run_paths(state["run_id"])
    tpath = rd / "transcript.json"
    body = {"run_id": state["run_id"], "approval_granted": False, "proposal_md_path": state.get("proposal_md_path")}
    tpath.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return {"transcript_path": str(tpath), "jira_projection": {"issues": [], "approval_granted": False}}


def compile_demo_graph(deps: GraphDeps, *, checkpointer: BaseCheckpointSaver | None = None):
    saver = checkpointer if checkpointer is not None else MemorySaver()
    sg = StateGraph(AgentState)

    sg.add_node("fetch_story_policy", lambda s: _fetch_policy(s, deps))
    sg.add_node("ingest_confluence", lambda s: _ingest_confluence(s, deps))
    sg.add_node("build_retrieval_query", _retrieval_query_node)
    sg.add_node("rag_retrieve", lambda s: _rag_retrieve(s, deps))

    sg.add_node("load_approve_stories", _load_approve_stories)

    sg.add_node("assemble_prompt", lambda s: _assemble_prompt(s, deps))
    sg.add_node("synthesize_llm", lambda s: _synthesize_llm(s, deps))
    sg.add_node("validate_stories", _validate_stories)
    sg.add_node("fail_hard", _fail_hard)

    sg.add_node("prepare_proposal", _prepare_proposal)

    sg.add_node("human_gate", _human_gate)

    sg.add_node("ledger_write", lambda s: _ledger_write(s, deps))
    sg.add_node("jira_publish", lambda s: _jira_sink(s, deps))
    sg.add_node("reject_finalize", _reject_finalize)

    sg.add_edge(START, "fetch_story_policy")
    sg.add_edge("fetch_story_policy", "ingest_confluence")
    sg.add_edge("ingest_confluence", "build_retrieval_query")
    sg.add_edge("build_retrieval_query", "rag_retrieve")

    def _route_after_rag(state: AgentState) -> Literal["load_approve_stories", "assemble_prompt"]:
        return "load_approve_stories" if state.get("approve_with_path") else "assemble_prompt"

    sg.add_conditional_edges(
        "rag_retrieve",
        _route_after_rag,
        {"load_approve_stories": "load_approve_stories", "assemble_prompt": "assemble_prompt"},
    )
    sg.add_edge("assemble_prompt", "synthesize_llm")
    sg.add_edge("synthesize_llm", "validate_stories")
    sg.add_edge("load_approve_stories", "prepare_proposal")

    def _wrapped_route(state: AgentState) -> RoutePostValidate:
        return _route_post_validate(state, deps)

    sg.add_conditional_edges(
        "validate_stories",
        _wrapped_route,
        {
            "assemble_prompt": "assemble_prompt",
            "prepare_proposal": "prepare_proposal",
            "fail_hard": "fail_hard",
        },
    )
    sg.add_edge("prepare_proposal", "human_gate")

    def _route_gate(state: AgentState) -> Literal["ledger_write", "reject_finalize"]:
        return "ledger_write" if state.get("approval_granted") else "reject_finalize"

    sg.add_conditional_edges(
        "human_gate",
        _route_gate,
        {"ledger_write": "ledger_write", "reject_finalize": "reject_finalize"},
    )
    sg.add_edge("ledger_write", "jira_publish")

    sg.add_edge("reject_finalize", END)
    sg.add_edge("fail_hard", END)
    sg.add_edge("jira_publish", END)

    return sg.compile(checkpointer=saver)
