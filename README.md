# Requirements generation agent demo (Python + LangGraph)

End-to-end **demo repo**: ingest a Confluence-storage PRD fixture, hydrate **StoryPolicy** governance JSON, simulate **organization RAG**, generate **bounded user stories**, pause on a **human approval gate** (unless explicitly bypassed), then mint rows in an in-memory **Jira** mock—all orchestrated through a **`StateGraph`**.

## Prerequisites

- **Python 3.11+** (managed via **`uv`** or your preferred runner)
- **`OPENAI_API_KEY`** when you want the hosted LLM structured-output path (**`ChatOpenAI.with_structured_output`**). Omit the key or pass **`--offline`** to reuse the deterministic PRD heuristic.

Clone / enter the checkout, then sync dependencies:

```bash
uv sync
```

## Scripts

| Command | Purpose |
| --- | --- |
| **`uv run mock-dev`** | Launches **`uvicorn`** on **`127.0.0.1:8877`** with policy + knowledge retrieval + Confluence-ish APIs + mocked Jira; serves a **browser Confluence hub** at **`/`**. |
| **`uv run demo`** | Opens **`MOCK_BASE_URL/`** (**`127.0.0.1:8877`** by default) in your browser — primary stakeholder entry. |
| **`uv run demo open-hub`** | Same as bare **`demo`** (explicit entry). |
| **`uv run demo pipeline`** | Terminal LangGraph run (same knobs as historic **`demo`**: interactive approval gate, **`--ci`**, **`--offline`**, etc.). |
| **`python -m reqs_agent_demo ...`** | Same Typer launcher as **`uv run demo`** (subcommands **`open-hub`**, **`pipeline`**). |

### Mock surface (single FastAPI stack)

Bundled mocks answer:

- **`GET /`** — Confluence-styled viewer for **`fixtures/confluence/demo-prd.json`** with an action bar (**Generate Jira stories**). Posted runs mirror **`demo pipeline --fixture-mode --ci`** (auto-approve; never use that combination for stakeholder sign-off recordings).
- **`POST /demo/generate-jira-stories`** — Form-triggered LangGraph invocation (`fixture-mode`, **`auto_approve=True`**). **Defaults to offline heuristic stories** unless the form submits **`use_openai=yes`** and **`OPENAI_API_KEY`** is set in the **`mock-dev`** process. Browsers **`303`** to **`/demo/issues?banner=generated`**. **`Accept`** without **`text/html`** yields JSON metadata instead of redirect.
- **`GET /v1/story-policy`** — emits **`fixtures/policy/demo.json`** (**`ETag`**, **`304`** aware).
- **`POST /v1/knowledge/retrieve`** — lexical / BM25-style ranking over **`fixtures/knowledge/**/*.md`** (YAML front matter encodes **`doc_type`**).
- **`GET /wiki/rest/api/content/demo-prd?expand=body.storage`** — serves **`fixtures/confluence/demo-prd.json`** (**“Building an authentication system”** canonical PRD JSON).
- **`POST /rest/api/3/issue`** — echoes summaries + persists **`_demo_requestedStatus`** alongside Atlassian-ish fields for **`fixtures/jira`-style parity.
- **`GET /rest/api/3/issue/{keyOrId}`** — returns JSON for API clients; **browsers** (**`Accept: text/html`**) get a Jira-styled story page. Append **`?format=json`** to force JSON, **`?format=html`** to force the page.
- **`GET /demo/issues`** — returns JSON (**`issues`** list) unless a browser asks for **`text/html`** (or **`?format=html`** / **`format=json`** override).

Production swap: reuse the **`httpx`** connectors (**`StoryPolicyClient`**, **`KnowledgeRetriever`**, **`ConfluenceConnector`**, **`JiraConnector`**) against real SaaS origins by changing base URLs/tokens (**see plan / ADRs referenced in corpuses**).

## CLI flags (highlights)

Run **`python -m reqs_agent_demo --help`** / **`demo pipeline --help`** for the full **`pipeline`** option table.

| Flag | When to use |
| --- | --- |
| **`--offline`** | Force deterministic heuristic stories (still honors StoryPolicy enums / AC cardinality). Skips **`OPENAI_API_KEY`** billing unless you intentionally unset offline + retain a key. |
| **`--ci`** (**`DEMO_AUTO_APPROVE`** env equivalent) | **UNSAFE DEMO BYPASS** — auto-approve backlog for CI smoke/tests. Never use for stakeholder recordings. |
| **`--fixture-mode`** | Read PRD + policy purely from **`fixtures/`**; **`KnowledgeRetriever(None)`** serves **`fixtures/knowledge/offline-pack.json`**. Still **`POST`**s mock Jira issues to **`MOCK_BASE_URL`**. |
| **`--approve-with path.json`** | Supply edited stories (**list** or **`{"stories": [...]}`**) that are **`coerce`‑validated** prior to **`ApprovedBacklog`** creation. Skips **`stdin`** interrupt. |
| **`--stream-events --ci`** | JSONL excerpt of **`graph.astream_events(..., version="v2")`**. **`--ci`** is required until interactive resume UX is threaded through streaming resumes. |

## Human-in-the-loop

1. **`prepare_approval_packet`** writes **`runs/<run_id>/proposal.md`** + **`proposal.json`** and embeds citations from the retrieved corpus.
2. **`human_gate`** calls **`interrupt(...)`**, surfacing payloads to the reviewer.
3. When running interactively (**TTY** available), **`typer.confirm`** issues **`Command(resume={"approve": …})`** to resume deterministically checkpointed graphs (**`MemorySaver`**).
4. Approval ledgers accumulate under **`approvals/<run_id>.json`** (hashes tie together policy blobs, retrieval dumps, reviewer metadata).

## Repository map

```
config/jira-field-map.json     # Demo Jira payloads: defaults (reporter, priority, labels, points, parent) + mock field IDs
fixtures/                      # Canonical PRD, policy enums, corpuses + offline corpus pack
prompt/rubric.md               # Soft guidance layered after StoryPolicy excerpts
runs/ … approvals/ …          # Ephemeral artefacts (ignored by `.gitignore`)
src/reqs_agent_demo/agent/    # Typed state + LangGraph compile + heuristic offline LLM path
```

## Environment variables

| Key | Meaning |
| --- | --- |
| **`MOCK_BASE_URL`** | Origin for mocks + Jira shim (default **`http://127.0.0.1:8877`). |
| **`REQS_AGENT_DEMO_ROOT`** | Override filesystem root locating **`fixtures/`** + **`config/`** (defaults to ancestor walk from package). |
| **`OPENAI_*` / **`OPENAI_MODEL`** | Passed through **`langchain-openai`**. |
| **`DEMO_AUTO_APPROVE`** | **`1/true/yes`** maps to **`--ci`** latch (unsafe). |
| **`POLICY_SERVICE_TOKEN`**, **`KNOWLEDGE_SERVICE_TOKEN`**, **`JIRA_MOCK_TOKEN`** | Bearer headers echoed by connector layers (optional for mocks). |
| **`KNOWLEDGE_TOP_K`** | Caps retrieval payloads (clients + mock both honor **`top_k`**). |

## Observability

- **`runs/<run>/retrieval.json`** — persists the **`ContextPack`** returned by retrieval.
- **`graph.astream_events`** can be streamed with **`--stream-events --ci`** for LangGraph timelines (node enter/leave, **`on_chat_model_end`**, etc.).
- Fatal validation loops serialize **`runs/<run>/transcript.failure.json`**.

## Safeguards

- Story governance is **never** inlined in Python literals—**`StoryPolicy` JSON drives Pydantic `create_model`/validators** rebuilt per run.
- **Multimodal diagram ingestion stays out-of-band for v1**: architecture fixtures carry textual C4 surrogates + optional reviewer-only **`asset_url`** pointers.
- **Auto-approve** paths are prominently labeled **`UNSAFE`** in code + docs to discourage accidental stakeholder misuse.
