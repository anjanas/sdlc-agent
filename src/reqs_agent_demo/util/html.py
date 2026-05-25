"""Confluence-ish HTML helpers."""

from __future__ import annotations

import re
from typing import Any

from bs4 import BeautifulSoup


def html_to_plain_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    return soup.get_text("\n")


def headings_breadcrumbs(html: str) -> list[tuple[int, str]]:
    soup = BeautifulSoup(html, "lxml")
    crumbs: list[tuple[int, str]] = []
    for tag in soup.find_all(["h1", "h2", "h3"]):
        level = int(tag.name[1])
        label = tag.get_text(" ", strip=True)
        if label:
            crumbs.append((level, label))
    return crumbs


_WS = re.compile(r"\s+")


def extract_keywords(text: str, *, limit: int = 48) -> list[str]:
    raw = [_WS.sub(" ", t).strip().lower() for t in re.split(r"[^\w/+.-]+", text) if t.strip()]
    seen: set[str] = set()
    out: list[str] = []
    for tok in raw:
        if len(tok) < 3 or tok in seen:
            continue
        seen.add(tok)
        out.append(tok)
        if len(out) >= limit:
            break
    return out


def heuristic_query(title: str, breadcrumbs: list[tuple[int, str]] | list[dict[str, Any]], plain: str) -> str:
    if breadcrumbs and isinstance(breadcrumbs[0], dict):
        crumbs = [f"{b.get('level')}:{b.get('title')}" for b in breadcrumbs]  # type: ignore[index]
    else:
        crumbs = [f"{lvl}:{label}" for lvl, label in breadcrumbs]  # type: ignore[assignment]
    parts = [title, *crumbs, " ".join(extract_keywords(plain))]
    return _WS.sub(" ", " | ".join(p for p in parts if p)).strip()

