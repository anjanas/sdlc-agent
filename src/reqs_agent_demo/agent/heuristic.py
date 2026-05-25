"""Offline story synthesis from PRD HTML."""

from __future__ import annotations

import itertools
from typing import Any

from bs4 import BeautifulSoup

from reqs_agent_demo.policy_models import StoryPolicyEnvelope


def _draw(counter: itertools.count, values: list[str]) -> str:
    idx = next(counter)
    return values[idx % len(values)]


def _truncate_story_description(text: str, max_chars: int) -> str:
    text = text.strip()
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    clip = max(0, max_chars - 1)
    return text[:clip].rstrip() + "…"


def _offline_story_description(
    *,
    section_title: str,
    headline_summary: str,
    prd_signals: list[str],
    max_chars: int,
) -> str:
    """Narrative Jira-style description stitched from neighbouring PRD lines (no templated placeholders)."""

    deduped: list[str] = []
    seen: set[str] = set()
    for raw in prd_signals:
        line = " ".join(raw.split()).strip()
        low = line.lower()
        if not line or low in seen:
            continue
        seen.add(low)
        deduped.append(line)
        if len(deduped) >= 10:
            break

    headline = headline_summary.strip()
    intro = (
        f'Delivers outcomes described under PRD subsection "{section_title.strip()}". '
        f'It concentrates on "{headline}" within the broader authentication initiative. '
        "Cross-cutting requirements called out elsewhere in the PRD—transport security, MFA posture, "
        "credential hashing, brute-force defenses, audited MFA changes, vault-backed secrets—remain "
        "implicit guardrails unless a reviewer formally narrows scope."
    )

    body_parts: list[str] = [intro]
    if deduped:
        body_parts.append("")
        body_parts.append("Captured PRD intent (quotes / paraphrases):")
        for item in deduped:
            body_parts.append(f"• {item}")
    body_parts.append("")
    body_parts.append(
        "Acceptance criteria state the behavioural proof points; QA should prioritize journey-level "
        "evidence referenced here when designing scenarios."
    )
    return _truncate_story_description("\n".join(body_parts), max_chars)


def deterministic_stories_from_prd(policy: dict[str, Any], *, html: str) -> dict[str, Any]:
    envelope = StoryPolicyEnvelope.model_validate(policy)
    prod_counter = itertools.count()
    stat_counter = itertools.count()
    product_lines = envelope.enums["productLine"]
    statuses = envelope.enums["requestedStatus"]
    max_ac = int(envelope.fields["acceptanceCriteria"]["maxItems"])
    min_ac = int(envelope.fields["acceptanceCriteria"]["minItems"])

    soup = BeautifulSoup(html, "lxml")
    stories: list[dict[str, Any]] = []
    desc_bounds = envelope.fields.get("description") or {}
    max_desc = int(desc_bounds.get("max", 4000))

    def flush_story(summary: str, bucket: list[str], section_title: str) -> None:
        s_bounds = envelope.fields["summary"]
        smin, smax = int(s_bounds["min"]), int(s_bounds["max"])
        sum_text = summary.strip()[:smax]
        while len(sum_text) < smin:
            sum_text = f"{sum_text} · backlog slice"

        source_pts = [b.strip("- •\t ").strip() for b in bucket if b.strip()]

        trimmed = list(source_pts)
        if len(trimmed) < min_ac:
            trimmed.extend(
                [
                    "Instrumentation emits auditable telemetry for QA evidence.",
                    "UX copy conforms to credential privacy guidance.",
                    "Regression validates rollback if deployment fails.",
                ]
            )
        trimmed = trimmed[:max_ac]
        if len(trimmed) < min_ac:
            raise RuntimeError("offline heuristic could not satisfy acceptanceCriteria bounds")

        description = _offline_story_description(
            section_title=section_title,
            headline_summary=sum_text,
            prd_signals=source_pts,
            max_chars=max_desc,
        )

        stories.append(
            {
                "summary": sum_text,
                "acceptanceCriteria": trimmed[:max_ac],
                "productLine": _draw(prod_counter, product_lines),
                "requestedStatus": _draw(stat_counter, statuses),
                "description": description,
                "linkedAdrIds": [],
            }
        )

    for h2 in soup.find_all("h2"):
        section_title = h2.get_text(" ", strip=True)
        bullets: list[str] = []
        journeys: list[str] = []

        for sib in h2.find_next_siblings():
            if sib.name == "h2":
                break
            if getattr(sib, "name", None) == "ul":
                for li in sib.find_all("li"):
                    text = li.get_text(" ", strip=True)
                    if not text:
                        continue
                    lowered = text.lower()
                    if lowered.startswith("journey"):
                        journeys.append(text)
                    else:
                        bullets.append(text)
            elif getattr(sib, "name", None) == "p":
                para = sib.get_text(" ", strip=True)
                if para:
                    bullets.append(para)

        for journey in journeys:
            flush_story(f"{section_title} — {journey}", [journey] + bullets, section_title)

        if bullets:
            flush_story(section_title, bullets, section_title)

    if not stories:
        raise RuntimeError("offline heuristic failed — check PRD headings in fixture HTML")

    return {"stories": stories}
