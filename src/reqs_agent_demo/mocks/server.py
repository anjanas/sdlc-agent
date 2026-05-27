"""Bundled demo HTTP mocks (policy, knowledge RAG, Confluence, Jira)."""

from __future__ import annotations

import hashlib
import html
import json
import logging
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import quote

import frontmatter as pyfm
import uvicorn
from fastapi import FastAPI, Form, Header, HTTPException, Query, Request
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from starlette.concurrency import run_in_threadpool
from starlette.responses import HTMLResponse, RedirectResponse, Response

from reqs_agent_demo.agent.graph import compile_demo_graph
from reqs_agent_demo.connectors.jira import ApprovedBacklog, ApprovedStory, build_jira_issue_posts
from reqs_agent_demo.paths import config_path, project_root
from reqs_agent_demo.pipeline_runner import (
    build_graph_deps_for_run,
    close_graph_clients,
    run_generation_invoke,
)

_REPO_ROOT = project_root()

_log = logging.getLogger(__name__)

_CORPUS: list[dict[str, Any]]
_IDF: dict[str, float]
_ISSUES: list[dict[str, Any]]
_ISSUE_COUNTER: int


def _tokenise(text: str) -> list[str]:
    return [t for t in re.split(r"[^\w/+.-]+", text.lower()) if len(t) > 2]


def _build_corpus(repo_root: Path) -> tuple[list[dict[str, Any]], dict[str, float]]:
    corpus: list[dict[str, Any]] = []
    kdir = repo_root / "fixtures" / "knowledge"
    for path in sorted(kdir.glob("*.md")):
        post = pyfm.loads(path.read_text())
        corpus.append(
            {
                "id": post.metadata.get("id") or path.stem,
                "doc_type": post.metadata.get("doc_type") or "other",
                "title": post.metadata.get("title") or path.stem,
                "anchor": post.metadata.get("anchor") or "",
                "asset_url": post.metadata.get("asset_url"),
                "body": post.content,
            }
        )
    texts = [_tokenise(f"{doc['title']} \n {doc['body']}") for doc in corpus]
    dfs: Counter[str] = Counter()
    for tokens in texts:
        dfs.update(set(tokens))
    n_docs = max(len(corpus), 1)
    idf: dict[str, float] = {
        tok: math.log((1 + n_docs) / (1 + freq)) + 1 for tok, freq in dfs.items()
    }
    for doc, tokens in zip(corpus, texts):
        freq = Counter(tokens)
        doc["tokens"] = tokens
        doc["freq"] = freq
    return corpus, idf


_CORPUS, _IDF = _build_corpus(_REPO_ROOT)
_ISSUES = []
_ISSUE_COUNTER = 1

# Shared across HTTP requests so LangGraph `interrupt` / `Command(resume=…)` can resume a run.
_DEMO_GRAPH_CHECKPOINTER = MemorySaver()
# Populated when generation pauses at the human gate; keyed by `run_id` / LangGraph `thread_id`.
_JIRA_APPROVAL_PREVIEW: dict[str, list[dict[str, Any]]] = {}


def reset_store() -> None:
    """Reset in-memory demo Jira backlog (mostly for tests/manual replay)."""

    global _ISSUE_COUNTER  # pylint: disable=global-statements
    _ISSUES.clear()
    _ISSUE_COUNTER = 1
    _JIRA_APPROVAL_PREVIEW.clear()


def _score(query: str) -> list[tuple[float, dict[str, Any]]]:
    scored: list[tuple[float, dict[str, Any]]] = []
    q_tokens = Counter(_tokenise(query))
    for doc in _CORPUS:
        s = 0.0
        for tok, qtf in q_tokens.items():
            if tok not in _IDF:
                continue
            dtf = doc["freq"].get(tok, 0)
            if dtf == 0:
                continue
            s += (1 + math.log(dtf)) * _IDF[tok] * (1 + math.log(qtf))
        scored.append((s, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored


def _issue_api_dict(stored: dict[str, Any]) -> dict[str, Any]:
    fields_src = stored.get("fields_echo") or {}
    description = fields_src.get("description")
    if description is None and isinstance(stored.get("accepted_body"), dict):
        description = stored["accepted_body"].get("fields", {}).get("description")
    return {
        "id": stored["id"],
        "key": stored["key"],
        "self": stored["self"],
        "fields": {
            **fields_src,
            "summary": fields_src.get("summary", ""),
            "description": description,
        },
        "_demo_requestedStatus": stored.get("_demo_requestedStatus"),
    }


def _client_wants_html(request: Request) -> bool:
    accept = (request.headers.get("accept") or "").lower()
    if "text/html" in accept:
        return True
    return False


def _format_priority_display(fields: dict[str, Any]) -> str:
    pri = fields.get("priority") or ""
    if isinstance(pri, dict):
        name = pri.get("name")
        return str(name) if name else json.dumps(pri)
    return str(pri)


def _format_reporter_display(fields: dict[str, Any]) -> str:
    reporter = fields.get("reporter")
    if reporter is None:
        return ""
    if isinstance(reporter, dict):
        return str(
            reporter.get("emailAddress")
            or reporter.get("displayName")
            or reporter.get("name")
            or json.dumps(reporter)
        )
    return str(reporter)


def _format_labels_display(fields: dict[str, Any]) -> str:
    labels = fields.get("labels") or []
    if isinstance(labels, list):
        return ", ".join(str(x) for x in labels if str(x))
    return str(labels)


def _format_parent_display(fields: dict[str, Any]) -> str:
    parent = fields.get("parent") or ""
    if isinstance(parent, dict):
        return str(parent.get("key") or json.dumps(parent))
    return str(parent) if parent else ""


def _format_story_points_display(fields: dict[str, Any]) -> str:
    if "storyPoints" in fields and fields["storyPoints"] is not None:
        return str(fields["storyPoints"])
    return ""


def _format_feature_parent_display(fields: dict[str, Any]) -> str:
    link = fields.get("featureParentKey")
    if link is None or link == "":
        return ""
    if isinstance(link, dict):
        return str(link.get("key") or link.get("issueKey") or json.dumps(link))
    return str(link)


def _render_jira_issue_page(api: dict[str, Any], stored: dict[str, Any], request: Request) -> str:
    fields = api.get("fields") or {}
    summary = html.escape(str(fields.get("summary") or ""))
    key = html.escape(str(api.get("key") or ""))
    iid = html.escape(str(api.get("id") or ""))
    status = html.escape(str(api.get("_demo_requestedStatus") or "Backlog"))

    it = fields.get("issuetype") or {}
    it_name = html.escape(str(it.get("name") if isinstance(it, dict) else "Story"))

    proj = fields.get("project") or {}
    proj_key = html.escape(str(proj.get("key") if isinstance(proj, dict) else "DEMO"))
    proj_name = html.escape(str(proj.get("name") if isinstance(proj, dict) else "Demo project"))

    desc_raw = fields.get("description")
    if desc_raw is None:
        desc_raw = ""
    desc_plain = str(desc_raw)
    desc_textarea_body = html.escape(desc_plain)

    issue_key_raw = str(api.get("key") or "")
    desc_save_action = html.escape(f"/demo/jira-issue/{issue_key_raw}/description")

    detail_rows: list[str] = []
    skip_known = frozenset(
        {
            "summary",
            "description",
            "issuetype",
            "project",
            "priority",
            "reporter",
            "labels",
            "parent",
            "productLine",
            "acceptanceCriteria",
            "storyPoints",
            "featureParentKey",
        }
    )
    for k, v in sorted(fields.items()):
        if k in skip_known or v is None:
            continue
        if isinstance(v, (dict, list)):
            v_display = json.dumps(v, indent=2)
        else:
            v_display = str(v)
        detail_rows.append(
            f"<tr><th>{html.escape(str(k))}</th><td><pre class='pre'>{html.escape(v_display)}</pre></td></tr>"
        )
    extra_fields = "\n".join(detail_rows) if detail_rows else ""

    rep_esc = html.escape(_format_reporter_display(fields) or "—")
    pri_esc = html.escape(_format_priority_display(fields) or "—")
    lab_esc = html.escape(_format_labels_display(fields) or "—")
    par_plain = _format_parent_display(fields)
    feat_plain = _format_feature_parent_display(fields)
    par_esc = html.escape(par_plain or "—")
    feat_esc = html.escape(feat_plain or "—")
    sp_esc = html.escape(_format_story_points_display(fields) or "—")

    json_href = f"{request.url.path}?format=json"
    all_href = "/demo/issues"

    description_locked = bool(stored.get("_demo_description_locked"))
    description_readonly_inner = html.escape(desc_plain).replace("\n", "<br />\n")

    if description_locked:
        desc_section = f"""          <div class="section-title">Description</div>
          <p class="desc-hint"><strong>Saved.</strong> This description cannot be edited in this mock viewer anymore.</p>
          <div class="description">{description_readonly_inner or "<em>No description</em>"}</div>
"""
    else:
        desc_section = f"""          <div class="section-title">Description <span style="font-weight:400;color:var(--text-subtle);text-transform:none;font-size:12px">(editable mock)</span></div>
          <p class="desc-hint">Edit below, then click <strong>Update</strong>. After save the description locks and this button disappears.</p>
          <form method="post" action="{desc_save_action}" class="desc-form">
            <textarea name="description" class="description-edit" rows="10"
              spellcheck="true" placeholder="Plain-text issue description">{desc_textarea_body}</textarea>
            <button type="submit" class="save-desc">Update</button>
          </form>
"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{key} · {summary[:80]}</title>
  <style>
    :root {{
      --jira-blue: #0052cc;
      --jira-blue-hover: #0747a6;
      --text: #172b4d;
      --text-subtle: #5e6c84;
      --border: #dfe1e6;
      --surface: #ffffff;
      --canvas: #f4f5f7;
      --success: #00875a;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen, Ubuntu, "Fira Sans",
        "Droid Sans", "Helvetica Neue", sans-serif;
      font-size: 14px;
      color: var(--text);
      background: var(--canvas);
      line-height: 1.45;
    }}
    .topbar {{
      background: var(--jira-blue);
      color: #fff;
      padding: 0 24px;
      height: 56px;
      display: flex;
      align-items: center;
      gap: 12px;
      box-shadow: 0 1px 3px rgba(0,0,0,.12);
    }}
    .topbar strong {{ font-weight: 600; letter-spacing: 0.02em; }}
    .topbar .badge {{
      background: rgba(255,255,255,.2);
      padding: 2px 8px;
      border-radius: 3px;
      font-size: 12px;
    }}
    .shell {{
      max-width: 1100px;
      margin: 0 auto;
      padding: 24px 16px 48px;
    }}
    .issue-card {{
      background: var(--surface);
      border-radius: 3px;
      box-shadow: 0 1px 1px rgba(9, 30, 66, 0.25);
      border: 1px solid var(--border);
      overflow: hidden;
    }}
    .issue-header {{
      padding: 20px 24px 16px;
      border-bottom: 1px solid var(--border);
    }}
    .issue-meta {{
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 10px 16px;
      margin-bottom: 12px;
      color: var(--text-subtle);
      font-size: 12px;
    }}
    .issue-meta a {{ color: var(--jira-blue); text-decoration: none; }}
    .issue-meta a:hover {{ text-decoration: underline; }}
    .type-pill {{
      background: #deebff;
      color: var(--jira-blue);
      font-weight: 600;
      padding: 2px 8px;
      border-radius: 3px;
      font-size: 11px;
      text-transform: uppercase;
    }}
    .status-pill {{
      background: #e3fcef;
      color: var(--success);
      font-weight: 600;
      padding: 2px 10px;
      border-radius: 3px;
      font-size: 12px;
      border: 1px solid #abf5d1;
    }}
    h1 {{
      margin: 0;
      font-size: 24px;
      font-weight: 500;
      color: var(--text);
      letter-spacing: -0.01em;
    }}
    .issue-body {{
      display: grid;
      grid-template-columns: 1fr 280px;
      gap: 0;
    }}
    @media (max-width: 900px) {{
      .issue-body {{ grid-template-columns: 1fr; }}
    }}
    .main {{
      padding: 20px 24px 28px;
      border-right: 1px solid var(--border);
    }}
    @media (max-width: 900px) {{
      .main {{ border-right: none; border-bottom: 1px solid var(--border); }}
    }}
    .sidebar {{
      padding: 20px 20px 28px;
      background: #fafbfc;
    }}
    .section-title {{
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      color: var(--text-subtle);
      margin: 20px 0 8px;
    }}
    .section-title:first-child {{ margin-top: 0; }}
    .description {{
      margin-top: 8px;
      padding: 12px 14px;
      background: var(--canvas);
      border-radius: 3px;
      border: 1px solid var(--border);
      min-height: 48px;
    }}
    .desc-form {{
      margin-top: 8px;
    }}
    .description-edit {{
      display: block;
      width: 100%;
      min-height: 140px;
      margin: 0 0 10px;
      padding: 12px 14px;
      font: inherit;
      line-height: 1.45;
      color: var(--text);
      background: #fff;
      border: 1px solid var(--border);
      border-radius: 3px;
      resize: vertical;
      box-sizing: border-box;
    }}
    .desc-form .save-desc {{
      background: var(--jira-blue);
      color: #fff;
      border: none;
      border-radius: 3px;
      padding: 8px 14px;
      font-weight: 600;
      cursor: pointer;
      font-size: 13px;
    }}
    .desc-form .save-desc:hover {{
      background: var(--jira-blue-hover);
    }}
    .desc-hint {{
      font-size: 12px;
      color: var(--text-subtle);
      margin: 0 0 8px;
    }}
    .detail-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }}
    .detail-table th {{
      text-align: left;
      vertical-align: top;
      padding: 8px 12px 8px 0;
      width: 38%;
      color: var(--text-subtle);
      font-weight: 600;
    }}
    .detail-table td {{
      padding: 8px 0;
      word-break: break-word;
    }}
    pre.pre {{
      margin: 0;
      white-space: pre-wrap;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      font-size: 11px;
    }}
    .footer-links {{
      margin-top: 16px;
      font-size: 12px;
      color: var(--text-subtle);
    }}
    .footer-links a {{ color: var(--jira-blue); }}
  </style>
</head>
<body>
  <header class="topbar">
    <strong>Jira</strong>
    <span class="badge">Mock · Requirements demo</span>
  </header>
  <div class="shell">
    <article class="issue-card">
      <div class="issue-header">
        <div class="issue-meta">
          <span class="type-pill">{it_name}</span>
          <span><a href="{html.escape(json_href)}">{key}</a></span>
          <span>· ID {iid}</span>
          <span>· <span class="status-pill">{status}</span></span>
        </div>
        <h1>{summary}</h1>
      </div>
      <div class="issue-body">
        <div class="main">
{desc_section}{f'<div class="section-title">Fields</div><table class="detail-table">{extra_fields}</table>' if extra_fields else ""}
          <div class="footer-links">
            <a href="{html.escape(json_href)}">View as JSON</a>
            &nbsp;·&nbsp;
            <a href="{html.escape(all_href)}">All mock issues</a>
          </div>
        </div>
        <aside class="sidebar">
          <div class="section-title">Details</div>
          <table class="detail-table">
            <tr><th>Project</th><td><strong>{proj_key}</strong> — {proj_name}</td></tr>
            <tr><th>Issue type</th><td>{it_name}</td></tr>
            <tr><th>Status</th><td>{status}</td></tr>
            <tr><th>Reporter</th><td>{rep_esc}</td></tr>
            <tr><th>Priority</th><td>{pri_esc}</td></tr>
            <tr><th>Labels</th><td>{lab_esc}</td></tr>
            <tr><th>Parent</th><td>{par_esc}</td></tr>
            <tr><th>Feature link</th><td>{feat_esc}</td></tr>
            <tr><th>Story points</th><td>{sp_esc}</td></tr>
          </table>
        </aside>
      </div>
    </article>
  </div>
</body>
</html>"""


def _demo_confluence_fixture() -> dict[str, Any]:
    path = _REPO_ROOT / "fixtures" / "confluence" / "demo-prd.json"
    return json.loads(path.read_text())


def _render_confluence_hub_page(data: dict[str, Any]) -> str:
    title = html.escape(str(data.get("title", "PRD")))
    space_obj = data.get("space") or {}
    space_name = html.escape(str(space_obj.get("name") or space_obj.get("key") or "Space"))
    version = html.escape(str((data.get("version") or {}).get("number") or "?"))
    body_html = (
        ((data.get("body") or {}).get("storage") or {}).get("value")
        if isinstance(data.get("body"), dict)
        else ""
    )
    if not isinstance(body_html, str):
        body_html = str(body_html)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title} · Confluence (mock)</title>
  <style>
    :root {{
      --cfs-navy: #0747a6;
      --cfs-navy-soft: #0052cc;
      --cfs-canvas: #f4f5f7;
      --cfs-sidebar: #f7f8f9;
      --cfs-text: #172b4d;
      --cfs-border: #dfe1e6;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
        Oxygen, Ubuntu, "Fira Sans", "Helvetica Neue", sans-serif;
      background: var(--cfs-canvas);
      color: var(--cfs-text);
      line-height: 1.5;
    }}
    header.atlas {{
      background: linear-gradient(90deg, var(--cfs-navy) 0%, var(--cfs-navy-soft) 55%, #0747c4 100%);
      color: #fff;
      min-height: 56px;
      display: flex;
      align-items: center;
      padding: 0 24px;
      box-shadow: 0 3px 5px rgba(9,30,66,.08);
    }}
    header.atlas .logo {{
      font-weight: 600;
      letter-spacing: 0.04em;
      font-size: 15px;
    }}
    header.atlas .pill {{
      margin-left: 12px;
      background: rgba(255,255,255,.18);
      border-radius: 3px;
      padding: 2px 8px;
      font-size: 11px;
    }}
    .layout {{
      display: grid;
      grid-template-columns: 260px minmax(0, 980px);
      gap: 0;
      max-width: 1380px;
      margin: 0 auto;
    }}
    aside.nav {{
      background: var(--cfs-sidebar);
      border-right: 1px solid var(--cfs-border);
      padding: 20px 16px 48px;
      font-size: 13px;
    }}
    aside.nav .section {{
      margin-top: 18px;
      color: #5e6c84;
      text-transform: uppercase;
      font-size: 11px;
      font-weight: 700;
      letter-spacing: .06em;
    }}
    aside.nav ul {{ list-style: none; padding: 0; margin: 10px 0 0 0; }}
    aside.nav li {{ margin-bottom: 6px; }}
    aside.nav li a {{
      color: var(--cfs-navy-soft);
      text-decoration: none;
    }}
    main.doc {{
      background: #fff;
      padding: 0 0 64px;
    }}
    .crumbs {{
      font-size: 12px;
      color: #5e6c84;
      padding: 16px 40px;
      border-bottom: 1px solid var(--cfs-border);
    }}
    .titlebar {{
      padding: 20px 40px 16px;
      border-bottom: 1px solid var(--cfs-border);
    }}
    h1 {{
      margin: 0;
      font-size: 29px;
      font-weight: 500;
      letter-spacing: -.01em;
    }}
    .meta {{
      margin-top: 8px;
      font-size: 12px;
      color: #5e6c84;
      display: flex;
      gap: 16px;
      flex-wrap: wrap;
      align-items: center;
    }}
    .action-bar {{
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 12px;
      padding: 16px 40px;
      background: var(--cfs-canvas);
      border-bottom: 1px solid var(--cfs-border);
    }}
    .action-bar button {{
      background: var(--cfs-navy-soft);
      color: #fff;
      border: none;
      border-radius: 3px;
      padding: 8px 16px;
      font-weight: 600;
      font-size: 14px;
      cursor: pointer;
      box-shadow: 0 1px 2px rgba(9,30,66,.08);
    }}
    .action-bar button:hover {{ background: #0747a6; }}
    .action-bar label {{
      display: inline-flex;
      gap: 6px;
      align-items: center;
      font-size: 13px;
      color: var(--cfs-text);
    }}
    .action-bar span.hint {{
      font-size: 12px;
      color: #5e6c84;
      flex-basis: 100%;
      margin-left: 0;
      margin-top: 4px;
    }}
    .wiki-content {{
      padding: 32px 40px 72px;
      max-width: 880px;
    }}
    .wiki-content :is(h1, h2, h3) {{
      font-weight: 500;
      color: var(--cfs-text);
    }}
    .wiki-content ul {{
      padding-left: 22px;
    }}
    .wiki-content li {{
      margin-bottom: 6px;
    }}
    footer.note {{
      font-size: 12px;
      color: #5e6c84;
      padding: 24px 40px;
      border-top: 1px solid var(--cfs-border);
    }}
    @media (max-width: 960px) {{
      .layout {{ grid-template-columns: 1fr; }}
      aside.nav {{ border-right: none; border-bottom: 1px solid var(--cfs-border); }}
    }}
  </style>
</head>
<body>
  <header class="atlas">
    <span class="logo">CONFLUENCE CLOUD · LOCAL MOCK</span>
    <span class="pill">{space_name}</span>
    <span class="pill">v{version}</span>
  </header>

  <div class="layout">
    <aside class="nav">
      <div class="section">Spaces</div>
      <ul>
        <li><a href="#">Demo Space · Engineering</a></li>
      </ul>
      <div class="section">Page tree</div>
      <ul>
        <li><strong>PRDs</strong></li>
        <li style="margin-left:10px;"><a href="#">{title}</a></li>
      </ul>
    </aside>

    <main class="doc">
      <div class="crumbs">Spaces / {space_name} / Requirements / Product</div>

      <div class="titlebar">
        <h1>{title}</h1>
        <div class="meta">
          <span>Restricted · Org requirement review</span>
          <span>Owner: Account Services Guild</span>
          <span>Updated (mock fixture)</span>
        </div>
      </div>

      <div class="action-bar">
        <form method="post" action="/demo/generate-jira-stories">
          <button type="submit">Generate Jira stories</button>
          <label>
            <input type="checkbox" name="use_openai" value="yes" />
            Use hosted OpenAI (needs a valid OPENAI_API_KEY in the requirements-generation-agent shell)
          </label>
          <span class="hint">
            Defaults to deterministic heuristic stories (no billing). Mirrors
            <code style="background:#eaeff5;padding:1px 4px;border-radius:2px;">uv run demo pipeline --fixture-mode --offline</code>
            unless you tick OpenAI above. You must confirm draft stories — the hub shows mocked Jira field payloads
            (summary, status, reporter, priorities, acceptance criteria, and more) before anything is POSTed.
          </span>
        </form>
      </div>

      <article class="wiki-content">
        {body_html}
      </article>

      <footer class="note">
        This page is bundled HTML from <code>fixtures/confluence/demo-prd.json</code> —
        styling approximates reader mode, not vendor pixels.
      </footer>
    </main>
  </div>
</body>
</html>"""


def _render_demo_issues_page(request: Request, *, banner: str | None = None) -> str:
    banner_html = ""
    if banner == "generated":
        banner_html = (
            '<div class="banner ok">Stories were approved and created in the mock Jira backlog — '
            "open keys below as Jira-styled HTML.</div>"
        )
    elif banner == "rejected":
        banner_html = (
            '<div class="banner warn">You cancelled approval — '
            "<strong>no</strong> mock Jira stories were created. Generate again anytime from the PRD page.</div>"
        )
    elif banner == "validation_failed":
        banner_html = (
            '<div class="banner bad">Story validation exhausted allowed repairs — '
            "check <code>runs/&lt;run&gt;/transcript.failure.json</code> in the checkout (requirements-generation-agent stderr for details)."
            "</div>"
        )

    rows: list[str] = []
    origin = html.escape(str(request.base_url).rstrip("/"))
    for issue in _ISSUES:
        api_body = _issue_api_dict(issue)
        key = html.escape(str(api_body.get("key") or ""))
        summary = html.escape(str((api_body.get("fields") or {}).get("summary") or ""))
        href = f"{origin}/rest/api/3/issue/{issue.get('key', '')}?format=html"
        rows.append(f"<tr><td><a href=\"{html.escape(href)}\">{key}</a></td><td>{summary}</td></tr>")

    tbody = "".join(rows) if rows else "<tr><td colspan=\"2\"><em>No issues yet.</em></td></tr>"

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width"/><title>Jira backlog (mock)</title>
<style>
body {{ font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; background:#f4f5f7; margin:0; color:#172b4d; }}
.top {{ background:#0052cc; color:#fff; padding:14px 20px; font-weight:600; }}
.shell {{ max-width:900px; margin:24px auto; background:#fff; border-radius:3px;
  box-shadow:0 1px 1px rgba(9,30,66,.15); overflow:hidden; border:1px solid #dfe1e6; }}
table {{ border-collapse:collapse; width:100%; font-size:14px; }}
th, td {{ border-bottom:1px solid #ebecf0; text-align:left; padding:10px 16px; }}
th {{ font-size:11px; text-transform:uppercase; letter-spacing:.04em; color:#5e6c84; background:#fafbfc; }}
a {{ color:#0052cc; text-decoration:none; }}
a:hover {{ text-decoration:underline; }}
.links {{ padding:14px 16px; font-size:13px; color:#5e6c84; }}
.banner {{ margin:14px auto 0; max-width:900px; padding:10px 14px; border-radius:3px;
  font-size:13px; font-weight:600; }}
.banner.ok {{ background:#e3fcef; color:#064; border:1px solid #abf5d1; }}
.banner.warn {{ background:#fffae6; color:#974f00; border:1px solid #ffe58f; }}
.banner.bad {{ background:#ffebe6; color:#5d1f1f; border:1px solid #ffbdad; }}
</style></head><body><div class="top">Demo Jira backlog </div>
{banner_html}
<div class="shell"><table><thead><tr><th>Key</th><th>Summary</th></tr></thead><tbody>{tbody}</tbody></table>
<div class="links"><a href="{origin}/">&larr; Back to mocked Confluence PRD</a> · JSON: <code>/demo/issues?format=json</code></div></div></body></html>"""


def _hub_jira_field_map() -> dict[str, Any]:
    return json.loads(config_path("jira-field-map.json").read_text())


def _hub_issue_preview_posts(outcome_bundle: dict[str, Any]) -> list[dict[str, Any]]:
    """POST bodies identical to `/rest/api/3/issue`, for reviewer preview."""
    rid = str(outcome_bundle.get("run_id") or "pending")
    stories_raw = outcome_bundle.get("validated_stories") or []
    backlog = ApprovedBacklog(run_id=rid, stories=[ApprovedStory.model_validate(s) for s in stories_raw])
    return build_jira_issue_posts(backlog, _hub_jira_field_map())


def _cf(fields: dict[str, Any], field_map: dict[str, Any], key: str) -> Any | None:
    cf = field_map.get("customFields") or {}
    id_key = cf.get(key)
    if not isinstance(id_key, str) or not id_key.strip():
        return None
    return fields.get(id_key)


def _render_jira_story_approval_page(request: Request, *, run_id: str, posts: list[dict[str, Any]]) -> str:
    origin = html.escape(str(request.base_url).rstrip("/"))
    fm = _hub_jira_field_map()

    rows: list[str] = []
    for idx, post in enumerate(posts):
        display_num = idx + 1
        fields_lo = dict(post.get("fields") or {})
        status_txt = html.escape(str(post.get("_demo_requestedStatus") or ""))
        summary = html.escape(str(fields_lo.get("summary") or "(no summary)"))

        proj = fields_lo.get("project") or {}
        pkey = proj.get("key") if isinstance(proj, dict) else ""
        pkey_esc = html.escape(str(pkey or ""))

        it = fields_lo.get("issuetype") or {}
        it_name = it.get("name") if isinstance(it, dict) else it
        it_esc = html.escape(str(it_name or "Story"))

        ac_raw = _cf(fields_lo, fm, "acceptanceCriteria")
        ac_esc = ""
        if ac_raw is not None:
            body = html.escape(str(ac_raw))
            ac_esc = f'<pre style="margin:10px 0 0;background:#f4f5f7;padding:10px;border-radius:3px;white-space:pre-wrap">{body}</pre>'

        pl_raw = _cf(fields_lo, fm, "productLine")
        pl_esc = html.escape(str(pl_raw) if pl_raw is not None else "")

        sp_raw = _cf(fields_lo, fm, "storyPoints")
        pts_esc = html.escape("" if sp_raw is None else str(sp_raw))

        pri_esc = html.escape(_format_priority_display(fields_lo))
        rep_esc = html.escape(_format_reporter_display(fields_lo))
        lab_esc = html.escape(_format_labels_display(fields_lo))
        par_esc = html.escape(_format_parent_display(fields_lo))

        desc_raw = fields_lo.get("description")
        if desc_raw is None:
            desc_raw = ""
        desc_for_ta = html.escape(str(desc_raw))

        dl = "".join(
            [
                "<dl>",
                f'<div class="kv"><dt>Story</dt><dd><strong>#{display_num}</strong></dd></div>',
                '<div class="kv"><dt>Summary</dt><dd><strong>', summary, "</strong></dd></div>",
                f'<div class="kv"><dt>Project key</dt><dd>{pkey_esc}</dd></div>',
                f'<div class="kv"><dt>Issue type</dt><dd>{it_esc}</dd></div>',
                f'<div class="kv"><dt>Workflow status </dt><dd>{status_txt}</dd></div>',
                f'<div class="kv"><dt>Priority</dt><dd>{pri_esc}</dd></div>',
                f'<div class="kv"><dt>Reporter</dt><dd>{rep_esc}</dd></div>',
                f'<div class="kv"><dt>Labels</dt><dd>{lab_esc}</dd></div>',
                f'<div class="kv"><dt>Story points</dt><dd>{pts_esc}</dd></div>',
                f'<div class="kv"><dt>Parent</dt><dd>{par_esc}</dd></div>',
                f'<div class="kv"><dt>Product line</dt><dd>{pl_esc}</dd></div>',
                '<div class="kv"><dt>Acceptance criteria</dt><dd>',
                ac_esc or "<em>(none)</em>",
                "</dd></div>",
                '<div class="kv"><dt>Description (editable)</dt><dd>',
                '<label for="desc_',
                str(idx),
                '" class="sr-only">Edit description for story ',
                str(display_num),
                "</label>",
                (
                    f'<textarea id="desc_{idx}" name="story_description_{idx}" rows="8" '
                    f'spellcheck="true" placeholder="Plain-text description (saved when you approve)" '
                    f'class="desc">{desc_for_ta}</textarea>'
                ),
                '<p class="field-hint">Merged into validated stories before <code>POST /rest/api/3/issue</code>.</p>',
                "</dd></div>",
                "</dl>",
            ]
        )
        rows.append(f'<section class="card"><h2>Story {display_num}</h2>{dl}</section>')

    cards = "".join(rows) if rows else '<p class="empty"><em>No preview payload — validate run_id and try generating again.</em></p>'
    run_esc = html.escape(run_id)
    story_count = len(posts)

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width"/>
<title>Approve Jira stories (mock)</title>
<style>
body {{ margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
  background:#f4f5f7; color:#172b4d; }}
.hdr {{ background:#0052cc; color:#fff; padding:14px 20px; font-weight:600; }}
.wrap {{ max-width:920px; margin:16px auto 40px; padding:0 16px; }}
.runid {{ font-size:13px; color:#5e6c84; margin:8px 0 16px; }}
.card {{ background:#fff; border:1px solid #dfe1e6; border-radius:3px; padding:14px 16px;
  margin-bottom:16px; box-shadow:0 1px 1px rgba(9,30,66,.06); }}
.card h2 {{ margin:0 0 12px; font-size:15px; color:#0747a6; }}
.kv {{ margin:8px 0; }}
dl {{ margin:0; }}
dt {{ font-size:11px; text-transform:uppercase; letter-spacing:.04em; color:#5e6c84; }}
dd {{ margin:2px 0 0 0; font-size:14px; }}
.desc {{ width:100%; box-sizing:border-box; font-size:14px; padding:10px 12px; margin-top:6px;
  min-height:100px; border:2px solid #0052cc; border-radius:3px; resize:vertical; font-family:inherit;
  line-height:1.45; background:#fff; }}
.field-hint {{ margin:6px 0 0; font-size:12px; color:#5e6c84; }}
.sr-only {{ position:absolute; width:1px; height:1px; padding:0; margin:-1px; overflow:hidden;
  clip:rect(0,0,0,0); white-space:nowrap; border:0; }}
.actions {{ margin-top:22px; display:flex; gap:12px; flex-wrap:wrap; align-items:center; }}
.note {{ margin-top:10px; font-size:13px; color:#5e6c84; }}
button {{ cursor:pointer; font:inherit; }}
.btn-submit {{ border:none; background:#0052cc; color:#fff; border-radius:3px;
  padding:10px 16px; font-weight:600; }}
.btn-cancel {{ border:1px solid #dfe1e6; background:#fff; border-radius:3px; padding:10px 16px; }}
a.back {{ font-size:14px; color:#0052cc; text-decoration:none; }}
.empty {{ padding:14px 0; }}
</style></head><body><div class="hdr">Approve mock Jira stories</div>
<div class="wrap">
  <div class="runid">Run Id: <code>{run_esc}</code></div>
  <form method="post" action="/demo/jira-approval/decision">
    <input type="hidden" name="run_id" value="{run_esc}" />
    <input type="hidden" name="story_count" value="{story_count}" />
    <p>Review the fields below. You can edit each <strong>description</strong> before the mock
    <code>POST /rest/api/3/issue</code> runs. Approve to create issues, or cancel.</p>
    {cards}
    <div class="actions">
      <button type="submit" name="decision" value="approve" class="btn-submit">Create stories in mock Jira</button>
      <button type="submit" name="decision" value="reject" class="btn-cancel">Cancel — do not create</button>
      <span class="note"><a class="back" href="{origin}/">&larr; Back to PRD</a></span>
    </div>
  </form>
</div>
</body></html>"""


def _generation_job(origin: str, use_openai: bool) -> dict[str, Any]:
    """LangGraph fixture run; pauses at human gate (`auto_approve=False`) until HTTP resume."""

    model_local = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    api_key_present = bool(os.getenv("OPENAI_API_KEY", "").strip())
    if use_openai and not api_key_present:
        raise ValueError(
            "OpenAI checkbox is checked but OPENAI_API_KEY is empty — "
            "export a key or leave the box unchecked for offline heuristic stories."
        )
    offline_llm = not use_openai

    outcome_bundle, deps_box, _, _ = run_generation_invoke(
        page_id="demo-prd",
        offline_llm=offline_llm,
        approve_path_txt=None,
        auto_approve=False,
        fixture_mode=True,
        mock_origin=origin.rstrip("/"),
        max_repairs=3,
        model=model_local,
        checkpointer=_DEMO_GRAPH_CHECKPOINTER,
    )

    paused = outcome_bundle.get("__interrupt__") or []

    if paused:
        rid = str(outcome_bundle.get("run_id") or "").strip()
        if not rid:
            raise RuntimeError("approval interrupt missing run_id")
        preview_posts = _hub_issue_preview_posts(outcome_bundle)
        _JIRA_APPROVAL_PREVIEW[rid] = preview_posts
        return {"status": "need_approval", "run_id": rid}

    close_graph_clients(deps_box)

    proj_err = str((outcome_bundle.get("jira_projection") or {}).get("error") or "")
    if proj_err == "validation_exhausted":
        return {"status": "validation_failed", "outcome_bundle": outcome_bundle}

    _log.warning(
        "generation finished without human-gate pause; keys=%s",
        sorted(outcome_bundle.keys()),
    )
    return {"status": "unexpected_complete", "outcome_bundle": outcome_bundle}


def _resume_story_approval_job(
    origin: str, run_id: str, approve: bool, story_descriptions: list[str] | None = None
) -> dict[str, Any]:
    model_local = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    deps_box = build_graph_deps_for_run(
        page_id="demo-prd",
        fixture_mode=True,
        mock_origin=origin.rstrip("/"),
        max_repairs=3,
        model=model_local,
    )
    runner_graph = compile_demo_graph(deps_box, checkpointer=_DEMO_GRAPH_CHECKPOINTER)
    cfg = {"configurable": {"thread_id": run_id}}
    resume_payload: dict[str, Any] = {"approve": approve}
    if approve and story_descriptions is not None:
        resume_payload["story_descriptions"] = story_descriptions

    try:
        outcome_bundle: dict[str, Any] = runner_graph.invoke(Command(resume=resume_payload), cfg)
        return outcome_bundle
    finally:
        close_graph_clients(deps_box)
        _JIRA_APPROVAL_PREVIEW.pop(run_id, None)


app = FastAPI(title="Reqs Agent Demo Unified Mocks", version="0.1")


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def confluence_hub() -> HTMLResponse:
    payload = _demo_confluence_fixture()
    return HTMLResponse(_render_confluence_hub_page(payload))


@app.post("/demo/generate-jira-stories")
async def generate_jira_from_hub(request: Request, use_openai: str | None = Form(default=None)):
    openai_requested = use_openai == "yes"
    wants_html_nav = _client_wants_html(request)
    base = str(request.base_url).rstrip("/")

    try:
        result = await run_in_threadpool(_generation_job, str(request.base_url), openai_requested)
    except ValueError as bad:
        _log.warning("%s", bad)
        raise HTTPException(status_code=400, detail=str(bad)) from bad
    except Exception as exc:
        _log.exception("Generate Jira stories failed")
        raise HTTPException(
            status_code=500,
            detail=(
                "Generate failed — see requirements-generation-agent stderr for traceback. Typical causes: bogus OPENAI_API_KEY "
                "with OpenAI toggled on, mocks not on 8877, or run graph errors. "
                f"Underlying: {exc!s}"
            ),
        ) from exc

    if result["status"] == "need_approval":
        rid = str(result["run_id"])
        if wants_html_nav:
            return RedirectResponse(url=f"/demo/jira-approval?run_id={quote(rid)}", status_code=303)

        approve_url = f"{base}/demo/jira-approval?run_id={quote(rid)}"
        return {
            "ok": True,
            "awaiting_approval": True,
            "run_id": rid,
            "approve_review_page_html": approve_url,
            "decision_endpoint": f"{base}/demo/jira-approval/decision",
        }

    if result["status"] == "validation_failed":
        if wants_html_nav:
            return RedirectResponse(url="/demo/issues?banner=validation_failed", status_code=303)
        return {
            "ok": False,
            "error": "validation_exhausted",
            "jira_projection": (result["outcome_bundle"].get("jira_projection")),
        }

    _log.error("hub generation completed without hitting human gate pause (%r)", result.get("status"))

    if wants_html_nav:
        return RedirectResponse(url="/demo/issues?banner=validation_failed", status_code=303)

    return {
        "ok": False,
        "error": "unexpected_generation_outcome",
        "detail": {"status": result.get("status")},
    }


@app.get("/demo/jira-approval")
def demo_jira_story_approval(
    request: Request,
    run_id: str = Query(description="Returned from `/demo/generate-jira-stories` interrupt."),
) -> HTMLResponse:
    rid = (run_id or "").strip()
    if not rid:
        raise HTTPException(status_code=400, detail="run_id query parameter required")

    model_local = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    origin = str(request.base_url).rstrip("/")

    entries = list(_JIRA_APPROVAL_PREVIEW.get(rid, []))
    if not entries:
        deps_box = build_graph_deps_for_run(
            page_id="demo-prd",
            fixture_mode=True,
            mock_origin=origin,
            max_repairs=3,
            model=model_local,
        )
        runner_graph = compile_demo_graph(deps_box, checkpointer=_DEMO_GRAPH_CHECKPOINTER)
        cfg = {"configurable": {"thread_id": rid}}
        try:
            snap = runner_graph.get_state(cfg)
        finally:
            close_graph_clients(deps_box)

        vals = getattr(snap, "values", None)
        if isinstance(vals, dict):
            rebound = {**vals, "run_id": rid}
            entries = _hub_issue_preview_posts(rebound)
            if entries:
                _JIRA_APPROVAL_PREVIEW[rid] = entries

        if not entries:
            raise HTTPException(
                status_code=404,
                detail=(
                    "No pending approval session for that run Id — submit "
                    "**Generate Jira stories** again from `/`."
                ),
            )

    return HTMLResponse(_render_jira_story_approval_page(request, run_id=rid, posts=entries))


@app.post("/demo/jira-approval/decision")
async def demo_jira_story_approval_decision(request: Request):
    form = await request.form()

    def _mp_str(field: str) -> str:
        raw = form.get(field)
        return raw.strip() if isinstance(raw, str) else ""

    def _mp_positive_int(field: str) -> int:
        raw = form.get(field)
        if not isinstance(raw, str):
            return 0
        try:
            return max(0, int(raw.strip() or "0"))
        except ValueError:
            return 0

    rid = _mp_str("run_id")
    decision = _mp_str("decision")
    approve = decision.lower() == "approve"
    wants_html = _client_wants_html(request)

    story_descriptions: list[str] | None = None
    if approve:
        n = _mp_positive_int("story_count")
        story_descriptions = [_mp_str(f"story_description_{i}") for i in range(n)]

    if not rid:
        raise HTTPException(status_code=400, detail="run_id required")

    try:
        outcome_bundle = await run_in_threadpool(
            _resume_story_approval_job, str(request.base_url), rid, approve, story_descriptions
        )
    except Exception as exc:
        _log.exception("Resume approval failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    projection = outcome_bundle.get("jira_projection") or {}
    issued = projection.get("jira_issues")

    issued_keys = [
        str(item["key"]) for item in (issued if isinstance(issued, list) else []) if isinstance(item, dict)
    ]

    base = str(request.base_url).rstrip("/")

    if wants_html:
        if approve and issued_keys:
            return RedirectResponse(url="/demo/issues?banner=generated", status_code=303)
        if not approve:
            return RedirectResponse(url="/demo/issues?banner=rejected", status_code=303)
        return RedirectResponse(url="/demo/issues?banner=validation_failed", status_code=303)

    ok = approve and len(issued_keys) > 0
    return {
        "ok": ok,
        "approved": approve,
        "issue_keys_created": issued_keys,
        "demo_issues_html": base + "/demo/issues",
        "demo_issues_json": base + "/demo/issues?format=json",
        "projection": projection,
    }


@app.get("/v1/story-policy")
def story_policy(if_none_match: str | None = Header(default=None, alias="If-None-Match")):
    demo_path = _REPO_ROOT / "fixtures" / "policy" / "demo.json"
    blob = demo_path.read_bytes()
    etag = hashlib.sha256(blob).hexdigest()
    if if_none_match and if_none_match.strip('"') == etag:
        raise HTTPException(status_code=304)
    rev = json.loads(blob.decode())["revision"]
    return Response(
        content=blob,
        media_type="application/json",
        headers={
            "ETag": etag,
            "Cache-Control": "private, max-age=60",
            "X-Story-Policy-Revision": rev,
        },
    )


@app.post("/v1/knowledge/retrieve")
async def retrieve(body: dict[str, Any]):
    query = str(body.get("query") or "")
    top_k = int(body.get("top_k") or 6)
    scored = _score(query)
    chunks: list[dict[str, Any]] = []
    for score, doc in scored[:top_k]:
        citations = [{"source_id": doc["id"], "heading": doc.get("anchor", "")}]
        excerpt = doc["body"].strip().replace("\n", " ")
        chunks.append(
            {
                "id": doc["id"],
                "doc_type": doc["doc_type"],
                "score": float(score or 0.001),
                "excerpt": excerpt[:1200],
                "citations": citations,
                **({"asset_url": doc["asset_url"]} if doc.get("asset_url") else {}),
            }
        )
    return {"chunks": chunks, "query": query}


@app.post("/demo/jira-issue/{issue_key}/description")
async def demo_update_mock_issue_description(issue_key: str, request: Request):
    """Persist edited description into the in-memory mock issue store (browser HTML UX)."""

    form = await request.form()
    raw = form.get("description")
    body = raw if isinstance(raw, str) else ""

    found = False
    for stored in _ISSUES:
        if stored.get("key") == issue_key or str(stored.get("id") or "") == issue_key:
            found = True
            fe = dict(stored.get("fields_echo") or {})
            trimmed = body.strip()
            fe["description"] = trimmed if trimmed else None
            stored["fields_echo"] = fe
            accepted = stored.get("accepted_body")
            if isinstance(accepted, dict):
                accepted_fields = dict(accepted.get("fields") or {})
                accepted_fields["description"] = fe["description"]
                accepted["fields"] = accepted_fields
                stored["accepted_body"] = accepted
            stored["_demo_description_locked"] = True
            break

    if not found:
        raise HTTPException(status_code=404, detail="issue not found in mock store")

    loc = quote(issue_key, safe="")
    return RedirectResponse(
        url=f"/rest/api/3/issue/{loc}?format=html",
        status_code=303,
    )


@app.get("/wiki/rest/api/content/{content_id}")
def confluence_content(content_id: str, expand: str = Query(default="body.storage")):
    _ = expand
    if content_id != "demo-prd":
        raise HTTPException(status_code=404, detail="unknown content id")
    path = _REPO_ROOT / "fixtures" / "confluence" / "demo-prd.json"
    return json.loads(path.read_text())


@app.get("/rest/api/3/issue/{issue_id_or_key}")
def get_issue(
    request: Request,
    issue_id_or_key: str,
    fmt: str | None = Query(
        default=None,
        alias="format",
        description="Use `json` for raw API response; `html` forces the story page.",
    ),
):
    """JSON for API tooling; Jira-like HTML when the browser asks for text/html."""

    for stored in _ISSUES:
        if stored.get("key") == issue_id_or_key or stored.get("id") == issue_id_or_key:
            api_body = _issue_api_dict(stored)
            if fmt and fmt.lower() == "json":
                return api_body
            if fmt and fmt.lower() == "html":
                return HTMLResponse(_render_jira_issue_page(api_body, stored, request))
            if _client_wants_html(request):
                return HTMLResponse(_render_jira_issue_page(api_body, stored, request))
            return api_body

    raise HTTPException(
        status_code=404,
        detail="Issue does not exist or the mock store was reset since creation.",
    )


@app.post("/rest/api/3/issue")
async def create_issue(request: Request):
    global _ISSUE_COUNTER  # pylint: disable=global-statements
    payload = await request.json()
    summary = payload.get("fields", {}).get("summary")
    if not isinstance(summary, str) or not summary.strip():
        raise HTTPException(status_code=400, detail="summary required")

    issue = {
        "key": f"DEMO-{_ISSUE_COUNTER}",
        "id": str(_ISSUE_COUNTER),
        "self": f"{request.base_url}rest/api/3/issue/DEMO-{_ISSUE_COUNTER}",
        "fields_echo": payload.get("fields"),
        "_demo_requestedStatus": payload.get("_demo_requestedStatus"),
        "accepted_body": payload,
    }
    _ISSUES.append(issue)
    _ISSUE_COUNTER += 1
    return {"key": issue["key"], "id": issue["id"], "self": issue["self"]}


@app.get("/demo/issues")
def demo_issues(
    request: Request,
    banner: str | None = Query(default=None, description='e.g. "generated" after hub action'),
    fmt: str | None = Query(
        default=None,
        alias="format",
        description="`json` for API shape; `html` forces backlog page.",
    ),
):
    payload = {"issues": list(_ISSUES)}
    if fmt and fmt.lower() == "json":
        return payload
    if fmt and fmt.lower() == "html":
        return HTMLResponse(_render_demo_issues_page(request, banner=banner))
    if _client_wants_html(request):
        return HTMLResponse(_render_demo_issues_page(request, banner=banner))
    return payload




def main():
    uvicorn.run("reqs_agent_demo.mocks.server:app", host="127.0.0.1", port=8877, reload=False, factory=False)
