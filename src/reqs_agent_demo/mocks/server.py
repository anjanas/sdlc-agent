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

import frontmatter as pyfm
import uvicorn
from fastapi import FastAPI, Form, Header, HTTPException, Query, Request
from starlette.concurrency import run_in_threadpool
from starlette.responses import HTMLResponse, RedirectResponse, Response

from reqs_agent_demo.paths import project_root
from reqs_agent_demo.pipeline_runner import close_graph_clients, run_generation_invoke

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


def reset_store() -> None:
    """Reset in-memory demo Jira backlog (mostly for tests/manual replay)."""

    global _ISSUE_COUNTER  # pylint: disable=global-statements
    _ISSUES.clear()
    _ISSUE_COUNTER = 1


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
    return str(link) if link not in (None, "") else ""


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
    description_html = html.escape(str(desc_raw)).replace("\n", "<br />\n")

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
          <div class="section-title">Description</div>
          <div class="description">{description_html or "<em>No description</em>"}</div>
          {f'<div class="section-title">Fields</div><table class="detail-table">{extra_fields}</table>' if extra_fields else ""}
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
            Use hosted OpenAI (needs a valid OPENAI_API_KEY in the mock-dev shell)
          </label>
          <span class="hint">
            Defaults to deterministic heuristic stories (no billing). Mirrors
            <code style="background:#eaeff5;padding:1px 4px;border-radius:2px;">uv run demo pipeline --fixture-mode --ci --offline</code>
            unless you tick OpenAI above. Runs are auto-approved for this demo hub only.
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
            '<div class="banner ok">Stories were generated via the mocked Confluence action bar — '
            "open keys below as Jira-styled HTML.</div>"
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
</style></head><body><div class="top">Demo Jira backlog (in-memory mock)</div>
{banner_html}
<div class="shell"><table><thead><tr><th>Key</th><th>Summary</th></tr></thead><tbody>{tbody}</tbody></table>
<div class="links"><a href="{origin}/">&larr; Back to mocked Confluence PRD</a> · JSON: <code>/demo/issues?format=json</code></div></div></body></html>"""


app = FastAPI(title="Reqs Agent Demo Unified Mocks", version="0.1")


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def confluence_hub() -> HTMLResponse:
    payload = _demo_confluence_fixture()
    return HTMLResponse(_render_confluence_hub_page(payload))


def _generation_job(origin: str, use_openai: bool) -> dict[str, Any]:
    """Blocking LangGraph fixture run (fixture_mode + auto-approve)."""

    model_local = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    deps_box: Any = None

    api_key_present = bool(os.getenv("OPENAI_API_KEY", "").strip())
    if use_openai and not api_key_present:
        raise ValueError(
            "OpenAI checkbox is checked but OPENAI_API_KEY is empty — "
            "export a key or leave the box unchecked for offline heuristic stories."
        )
    offline_llm = not use_openai

    try:
        outcome_bundle, deps_box, _, _ = run_generation_invoke(
            page_id="demo-prd",
            offline_llm=offline_llm,
            approve_path_txt=None,
            auto_approve=True,
            fixture_mode=True,
            mock_origin=origin.rstrip("/"),
            max_repairs=3,
            model=model_local,
        )
        return outcome_bundle
    finally:
        if deps_box is not None:
            close_graph_clients(deps_box)


@app.post("/demo/generate-jira-stories")
async def generate_jira_from_hub(request: Request, use_openai: str | None = Form(default=None)):
    openai_requested = use_openai == "yes"
    wants_html_nav = _client_wants_html(request)

    try:
        await run_in_threadpool(_generation_job, str(request.base_url), openai_requested)
    except ValueError as bad:
        _log.warning("%s", bad)
        raise HTTPException(status_code=400, detail=str(bad)) from bad
    except Exception as exc:
        _log.exception("Generate Jira stories failed")
        raise HTTPException(
            status_code=500,
            detail=(
                "Generate failed — see mock-dev stderr for traceback. Typical causes: bogus OPENAI_API_KEY "
                "with OpenAI toggled on, mocks not on 8877, or run graph errors. "
                f"Underlying: {exc!s}"
            ),
        ) from exc

    if wants_html_nav:
        return RedirectResponse(url="/demo/issues?banner=generated", status_code=303)

    base = str(request.base_url).rstrip("/")
    return {
        "ok": True,
        "demo_issues_html": base + "/demo/issues",
        "demo_issues_json": base + "/demo/issues?format=json",
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
