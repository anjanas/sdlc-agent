"""Convenience factory wiring for mocks vs future cloud backends."""

from __future__ import annotations

from reqs_agent_demo.connectors.confluence import ConfluenceConnector
from reqs_agent_demo.connectors.jira import JiraConnector
from reqs_agent_demo.connectors.knowledge import KnowledgeRetriever
from reqs_agent_demo.connectors.policy_client import StoryPolicyClient


def create_demo_clients(
    confluence_base_url: str | None,
    jira_base_url: str,
    policy_base_url: str | None,
    knowledge_base_url: str | None,
    *,
    token_confluence: str | None = None,
    token_jira: str | None = None,
    token_policy: str | None = None,
    token_knowledge: str | None = None,
    tenant_policy: str | None = None,
):
    cc = (
        ConfluenceConnector(confluence_base_url, token=token_confluence) if confluence_base_url else None
    )

    jc = JiraConnector(jira_base_url, token=token_jira)

    pol = StoryPolicyClient(policy_base_url, token=token_policy, tenant=tenant_policy) if policy_base_url else None

    kr = KnowledgeRetriever(knowledge_base_url, token=token_knowledge)
    return cc, pol, kr, jc


def create_atlassian_clients(site_url: str, token: str) -> tuple[ConfluenceConnector, JiraConnector]:
    cf = site_url.strip().rstrip("/")
    return ConfluenceConnector(cf, token=token), JiraConnector(cf, token=token)
