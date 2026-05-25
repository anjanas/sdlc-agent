---
id: adr-042-password-hashing
doc_type: adr
title: "ADR-042 — Password hashing with Argon2id"
anchor: adr-042
---

# Decision

Standardize credential hashing on **Argon2id** with explicit memory/time params managed by Secrets Manager rotations.

## Consequences

- Login services must expose metrics for verify latency budgeting.
- Password reset completions re-hash via the same verifier contract.
