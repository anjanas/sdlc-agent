"""Jira-shaped writer."""

from __future__ import annotations

from typing import Any

import httpx

from pydantic import BaseModel, ConfigDict, Field


class ApprovedStory(BaseModel):
    model_config = ConfigDict(extra="ignore")

    summary: str
    acceptanceCriteria: list[str]
    productLine: str
    requestedStatus: str
    description: str | None = None
    linkedAdrIds: list[str] | None = None

    reporter_email: str | None = None
    priority: str | None = Field(
        default=None,
        description="Jira Priority name when overriding field-map default",
    )
    labels: list[str] | None = None
    story_points: int | float | None = Field(
        default=None,
        ge=0,
        description="Numeric story points; falls back to jira-field-map defaults.storyPoints",
    )
    parent_issue_key: str | None = Field(
        default=None,
        description="Parent Feature/Epic issue key shown as parent/feature link in Jira mocks",
    )


class ApprovedBacklog(BaseModel):
    run_id: str
    approved_by_id: str | None = None
    approved_by_email: str | None = None
    stories: list[ApprovedStory]


class JiraConnector:
    def __init__(self, base_url: str, *, token: str | None = None, timeout: float = 45.0):
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self._client = httpx.Client(base_url=base_url.rstrip("/"), headers=headers, timeout=timeout)
        self._field_map_path: Any = None

    def close(self) -> None:
        self._client.close()

    def create_stories(self, backlog: ApprovedBacklog, field_map: dict[str, Any]) -> list[dict[str, Any]]:
        custom_fields_map = dict(field_map.get("customFields", {}))
        defaults = dict(field_map.get("defaults") or {})
        results: list[dict[str, Any]] = []
        proj = field_map["projectKey"]
        issue_type = field_map["issueTypeName"]

        ac_field = custom_fields_map.get("acceptanceCriteria") or ""
        prod_field = custom_fields_map.get("productLine") or ""
        sp_field = custom_fields_map.get("storyPoints") or ""
        feature_parent_cf = (
            custom_fields_map.get("featureParentKey") or custom_fields_map.get("epicLink") or ""
        )

        reporter_default = defaults.get("reporterEmail") or "janedoe@demo.com"
        priority_default = defaults.get("priority") or "Medium"
        labels_default = [str(l) for l in (defaults.get("labels") or []) if str(l)]
        points_default_raw = defaults.get("storyPoints", 3)
        try:
            points_default_f: int | float = float(points_default_raw)
            if points_default_f == int(points_default_f):
                points_default_f = int(points_default_f)
        except (TypeError, ValueError):
            points_default_f = 3
        parent_default = defaults.get("parentIssueKey")

        def _merged_labels(extra: list[str] | None) -> list[str]:
            seen: dict[str, None] = {}
            merged: list[str] = []
            for lab in [*labels_default, *(extra or [])]:
                lowered = lab.strip()
                if not lowered or lowered.casefold() in seen:
                    continue
                seen[lowered.casefold()] = None
                merged.append(lab.strip())
            return merged

        for story in backlog.stories:
            plain_ac = "\n".join(f"- {b}" for b in story.acceptanceCriteria)

            reporter_email = story.reporter_email or reporter_default
            priority_name = story.priority or priority_default
            merged_lb = _merged_labels(story.labels)
            parent_key_raw = story.parent_issue_key if story.parent_issue_key is not None else parent_default

            pts = story.story_points if story.story_points is not None else points_default_f

            fields: dict[str, Any] = {
                "project": {"key": proj},
                "summary": story.summary,
                "issuetype": {"name": issue_type},
                "description": story.description or plain_ac[:400],
                "priority": {"name": priority_name},
                "labels": merged_lb,
                "reporter": {"emailAddress": reporter_email},
            }
            payload: dict[str, Any] = {
                "fields": fields,
                "_demo_requestedStatus": story.requestedStatus,
            }

            parent_key_trim = (
                parent_key_raw.strip() if isinstance(parent_key_raw, str) and parent_key_raw.strip() else None
            )

            if parent_key_trim:
                fields["parent"] = {"key": parent_key_trim}
                if feature_parent_cf:
                    fields[feature_parent_cf] = parent_key_trim

            if prod_field:
                fields[prod_field] = story.productLine
            if ac_field:
                fields[ac_field] = plain_ac

            if sp_field and pts is not None:
                fields[sp_field] = pts

            resp = self._client.post("/rest/api/3/issue", json=payload)
            resp.raise_for_status()
            results.append(resp.json())
        return results
