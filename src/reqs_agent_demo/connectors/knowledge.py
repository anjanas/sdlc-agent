"""Knowledge / RAG retriever shim."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from reqs_agent_demo.paths import fixtures_path


@dataclass
class RetrievalChunk:
    id: str
    doc_type: str
    score: float
    excerpt: str
    citations: list[dict[str, str]]


@dataclass
class ContextPack:
    query: str
    chunks: list[RetrievalChunk]

    def to_system_snippet(self) -> str:
        buckets: dict[str, list[str]] = {
            "Story templates retrieved": [],
            "Acceptance criteria playbook": [],
            "Relevant ADRs": [],
            "Architecture context": [],
        }
        for ch in sorted(self.chunks, key=lambda c: (-c.score, c.doc_type)):
            label = {"story_template": "Story templates retrieved"}.get(
                ch.doc_type, None
            ) or {"ac_playbook": "Acceptance criteria playbook"}.get(ch.doc_type)  # pylint: disable=line-too-long
            if label is None:
                if ch.doc_type == "adr":
                    label = "Relevant ADRs"
                elif ch.doc_type == "architecture_diagram_text":
                    label = "Architecture context"
                else:
                    label = "Other knowledge"
            cite = "; ".join(
                f"{c.get('source_id','')}#{c.get('heading','')}" for c in ch.citations
            )
            body = ch.excerpt.strip().replace("\n", " ")
            buckets.setdefault(label, []).append(f"- (**{ch.id}**, score={ch.score:.3f}; {cite}) {body}")

        segments: list[str] = []
        for title in [
            "Story templates retrieved",
            "Acceptance criteria playbook",
            "Relevant ADRs",
            "Architecture context",
            "Other knowledge",
        ]:
            rows = buckets.get(title, [])
            if not rows:
                continue
            segments.append(f"### {title}\n" + "\n".join(rows))
        return "\n\n".join(segments)

    def list_citations(self) -> list[str]:
        out: list[str] = []
        for ch in self.chunks:
            for cite in ch.citations:
                out.append(
                    f"{ch.id}:{ch.doc_type} -> {cite.get('source_id','')}#{cite.get('heading','')}"
                )
        return out

    def to_serialisable(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "chunks": [
                {
                    "id": ch.id,
                    "doc_type": ch.doc_type,
                    "score": ch.score,
                    "excerpt": ch.excerpt,
                    "citations": ch.citations,
                }
                for ch in self.chunks
            ],
        }


class KnowledgeRetriever:
    def __init__(
        self,
        base_url: str | None,
        *,
        token: str | None = None,
        top_k: int = 8,
        cache_ttl_seconds: float = 15.0,
        offline_fixture: Path | None = None,
    ):
        hdrs = {"Accept": "application/json", "Content-Type": "application/json"}
        if token:
            hdrs["Authorization"] = f"Bearer {token}"
        self._top_k = top_k
        default_offline = fixtures_path("knowledge", "offline-pack.json")
        self._offline_fixture = offline_fixture if offline_fixture is not None else default_offline
        self._ttl = cache_ttl_seconds
        self._cache: dict[int, tuple[float, ContextPack]] = {}
        self._client: httpx.Client | None = None
        base = base_url.rstrip("/") if base_url else None
        if base:
            self._client = httpx.Client(base_url=base, headers=hdrs, timeout=30.0)

    def close(self) -> None:
        if self._client:
            self._client.close()

    def retrieve(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
        categories: list[str] | None = None,
        top_k: int | None = None,
    ) -> ContextPack:
        key_payload = "|".join(
            (
                query,
                json.dumps(filters or {}, sort_keys=True),
                json.dumps(categories or [], sort_keys=True),
                str(top_k or self._top_k),
            )
        ).encode()
        cache_key = hash(key_payload)
        now = time.time()
        cached = self._cache.get(cache_key)
        if cached and now - cached[0] < self._ttl:
            return cached[1]

        if not self._client:
            path = Path(self._offline_fixture)
            data = json.loads(path.read_text())
            chunks = [
                RetrievalChunk(
                    id=c["id"],
                    doc_type=c["doc_type"],
                    score=float(c.get("score", 0.0)),
                    excerpt=c["excerpt"],
                    citations=c.get("citations", []) or [],
                )
                for c in data
            ]
            pack = ContextPack(query=query, chunks=chunks)
            self._cache[cache_key] = (now, pack)
            return pack

        tk = top_k or self._top_k
        assert self._client is not None
        body = {"query": query, "filters": filters or {}, "categories": categories or [], "top_k": tk}
        resp = self._client.post("/v1/knowledge/retrieve", json=body)
        resp.raise_for_status()
        payload = resp.json()
        chunks: list[RetrievalChunk] = []
        for c in payload.get("chunks", []):
            chunks.append(
                RetrievalChunk(
                    id=c["id"],
                    doc_type=c["doc_type"],
                    score=float(c.get("score", 0.0)),
                    excerpt=c["excerpt"],
                    citations=c.get("citations") or [],
                )
            )
        pack = ContextPack(query=query, chunks=chunks)
        self._cache[cache_key] = (now, pack)
        return pack
