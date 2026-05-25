"""Pydantic helpers for governance policy JSON."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model, field_validator


class StoryPolicyEnvelope(BaseModel):
    revision: str
    fields: dict[str, Any]
    enums: dict[str, list[str]]
    requiredFields: dict[str, bool] = Field(default_factory=dict)
    prompt: dict[str, Any] = Field(default_factory=dict)


def _build_validated_story_class(policy_raw: dict[str, Any]) -> type[BaseModel]:
    fp = StoryPolicyEnvelope.model_validate(policy_raw)
    sums = fp.fields["summary"]
    ac = fp.fields["acceptanceCriteria"]
    product_lines = list(fp.enums["productLine"])
    statuses = list(fp.enums["requestedStatus"])
    s_min = int(sums["min"])
    s_max = int(sums["max"])
    ac_min = int(ac["minItems"])
    ac_max = int(ac["maxItems"])

    class ValidatedStory(BaseModel):  # type: ignore[misc, valid-type]
        model_config = ConfigDict(extra="forbid")

        summary: str
        acceptanceCriteria: list[str]
        productLine: str
        requestedStatus: str
        description: str | None = None
        linkedAdrIds: list[str] | None = None

        @field_validator("summary")
        @classmethod
        def validate_summary(cls, v: str) -> str:
            t = v.strip()
            if not (s_min <= len(t) <= s_max):
                raise ValueError(f"summary length must be within {s_min}..{s_max}")
            return t

        @field_validator("acceptanceCriteria")
        @classmethod
        def validate_ac(cls, v: list[str]) -> list[str]:
            items = [b.strip() for b in v if isinstance(b, str) and b.strip()]
            if not (ac_min <= len(items) <= ac_max):
                raise ValueError(f"acceptanceCriteria must have between {ac_min} and {ac_max} bullets")
            return items

        @field_validator("productLine")
        @classmethod
        def validate_product(cls, v: str) -> str:
            candidate = v.strip()
            if candidate not in product_lines:
                allowed = ", ".join(product_lines)
                raise ValueError(f"productLine must be one of: {allowed}")
            return candidate

        @field_validator("requestedStatus")
        @classmethod
        def validate_status(cls, v: str) -> str:
            candidate = v.strip()
            if candidate not in statuses:
                allowed = ", ".join(statuses)
                raise ValueError(f"requestedStatus must be one of: {allowed}")
            return candidate

    return ValidatedStory


def stories_bundle_model(policy: dict[str, Any]) -> tuple[type[BaseModel], type[BaseModel]]:
    Story = _build_validated_story_class(policy)
    Bundle = create_model(
        "StoriesEnvelope",
        stories=(list[Story], Field(..., min_length=1)),
        __base__=BaseModel,
    )
    return Story, Bundle


def coerce_stories(
    policy: dict[str, Any], payload: dict[str, Any]
) -> tuple[list[dict[str, Any]] | None, list[ValidationError]]:
    _, Bundle = stories_bundle_model(policy)
    errs: list[ValidationError] = []
    try:
        bundle = Bundle.model_validate(payload)
    except ValidationError as e:
        errs.append(e)
        return None, errs
    return [s.model_dump() for s in bundle.stories], errs  # type: ignore[attr-defined]
