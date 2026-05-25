"""Story governance policy client."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx

from reqs_agent_demo.policy_models import StoryPolicyEnvelope


def policy_prompt_fragments(env: StoryPolicyEnvelope) -> list[str]:
    """Build governance prompt lines from envelope (pure; no HTTP)."""

    fragments: list[str] = [
        f"Policy revision `{env.revision}` governs enumerated fields:",
        "**productLine enums:** "
        + ", ".join(env.enums.get("productLine", [])),
        "**requestedStatus enums:** " + ", ".join(env.enums.get("requestedStatus", [])),
        "**summary constraints:** "
        + f"{env.fields['summary']['min']}-{env.fields['summary']['max']} chars",
        "**acceptanceCriteria length:** "
        + f"{env.fields['acceptanceCriteria']['minItems']}-"
        + f"{env.fields['acceptanceCriteria']['maxItems']} bullets",
    ]
    for block in env.prompt.get("hardSections", []) or []:
        if isinstance(block, str):
            fragments.append(block)
    return fragments


class StoryPolicyClient:
    """Fetches authoritative JSON defining enums and field governance."""

    def __init__(
        self,
        base_url: str,
        *,
        token: str | None = None,
        tenant: str | None = None,
        timeout: float = 30.0,
    ):
        self._base = base_url.rstrip("/")
        hdrs: dict[str, str] = {"Accept": "application/json"}
        if token:
            hdrs["Authorization"] = f"Bearer {token}"
        self._tenant = tenant
        self._client = httpx.Client(headers=hdrs, timeout=timeout)
        self._etag_cache: dict[str, str] = {}

    def close(self) -> None:
        self._client.close()

    def fetch_story_policy(self) -> tuple[StoryPolicyEnvelope, dict[str, str]]:
        params: dict[str, str] = {}
        if self._tenant:
            params["tenant"] = self._tenant
        url = f"{self._base}/v1/story-policy"
        headers: dict[str, str] = {}
        key = ":".join(sorted(params.items()))
        if last := self._etag_cache.get(key):
            headers["If-None-Match"] = last
        resp = self._client.get(url, params=params or None, headers=headers)
        if resp.status_code == httpx.codes.NOT_MODIFIED:
            raise RuntimeError("304 Not Modified but no cached body in-memory (demo)")
        resp.raise_for_status()
        etag = resp.headers.get("ETag") or ""
        if etag:
            self._etag_cache[key] = etag
        envelope = StoryPolicyEnvelope.model_validate(resp.json())
        return envelope, dict(resp.headers)

    def policy_to_prompt_fragments(self, policy: StoryPolicyEnvelope) -> list[str]:
        return policy_prompt_fragments(policy)


def load_policy_from_file(path: Path | str) -> StoryPolicyEnvelope:
    payload = json.loads(Path(path).read_text())
    return StoryPolicyEnvelope.model_validate(payload)


def envelope_to_primitive(env: StoryPolicyEnvelope) -> dict[str, Any]:
    return env.model_dump()
