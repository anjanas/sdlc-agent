"""Resolve repository root containing `fixtures/`."""

from __future__ import annotations

import os
from pathlib import Path

_ENV_ROOT = "REQS_AGENT_DEMO_ROOT"


def project_root() -> Path:
    if env := os.getenv(_ENV_ROOT):
        return Path(env).expanduser().resolve()
    here = Path(__file__).resolve()
    for p in [here.parent, *here.parents]:
        if (p / "fixtures").exists() and (p / "config").exists():
            return p
    return Path.cwd()


def fixtures_path(*parts: str) -> Path:
    return project_root().joinpath("fixtures", *parts)


def prompt_path(name: str) -> Path:
    return project_root().joinpath("prompt", name)


def config_path(name: str) -> Path:
    return project_root().joinpath("config", name)
