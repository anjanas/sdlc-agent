"""Confluence Cloud shaped reader."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx

from reqs_agent_demo.util.html import headings_breadcrumbs, html_to_plain_text


class ConfluenceConnector:
    def __init__(
        self,
        base_url: str,
        *,
        token: str | None = None,
        timeout: float = 30.0,
    ):
        headers: dict[str, str] = {"Accept": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self._base = base_url.rstrip("/")
        self._client = httpx.Client(headers=headers, timeout=timeout)

    def close(self) -> None:
        self._client.close()

    def get_page(self, page_id: str) -> dict[str, Any]:
        url = (
            f"{self._base}/wiki/rest/api/content/"
            + f"{page_id}?expand=body.storage,space,version"
        )
        resp = self._client.get(url)
        resp.raise_for_status()
        data = resp.json()
        html = data["body"]["storage"]["value"]
        title_local = data.get("title", "")
        breadcrumbs_pairs = headings_breadcrumbs(html)
        breadcrumbs_serial = [{"level": lvl, "title": lbl} for lvl, lbl in breadcrumbs_pairs]
        plaintext = html_to_plain_text(html)
        return {
            "id": page_id,
            "title": title_local,
            "space_key": data.get("space", {}).get("key"),
            "html": html,
            "plain_text": plaintext,
            "breadcrumbs": breadcrumbs_serial,
        }


def offline_confluence_fixture(page_path: Path | str) -> dict[str, Any]:
    """Load serialized JSON resembling Confluence REST for offline ingestion."""
    import json

    path_obj = Path(page_path)
    data = json.loads(path_obj.read_text())
    html_local = data["body"]["storage"]["value"]
    title_local = data.get("title", "")
    breadcrumbs_pairs = headings_breadcrumbs(html_local)
    breadcrumbs_serial_local = [{"level": lvl, "title": lbl} for lvl, lbl in breadcrumbs_pairs]
    plaintext_local = html_to_plain_text(html_local)
    return {
        "id": data["id"],
        "title": title_local,
        "space_key": data.get("space", {}).get("key"),
        "html": html_local,
        "plain_text": plaintext_local,
        "breadcrumbs": breadcrumbs_serial_local,
    }
