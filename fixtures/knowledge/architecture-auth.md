---
id: c4-auth-flow-text
doc_type: architecture_diagram_text
title: "Auth subsystem narrative"
anchor: architecture-auth
asset_url: "https://example.invalid/diagrams/auth-c4-placeholder.svg"
---

Components: **Auth Gateway** terminates TLS and applies WAF hints; forwards to **Session Service**
for credential + MFA choreography; persists identities via **Identity store** guarded by VPC policies.
Password reset correlates outbound mailer tokens with hashed lookup tables mirrored in Session Service.
