# Soft rubric (non-authoritative tone)

Authoritative enums, bounds, and required fields arrive from **`StoryPolicy`** retrieved at runtime — do not contradict them.

- Write **behavioral acceptance criteria**: verifiable outcomes, not implementation steps.
- Keep **summaries concise** as `Story`-style headlines; reserve detail for **`description`** and AC bullets.
- Prefer **parity with retrieved org templates** surfaced under “Story templates” and “Acceptance criteria playbook” excerpts.
- When architecture or ADRs appear in citations, cite their identifiers in **`description`** (and **`linkedAdrIds`** only if explicitly allowed later).
- For auth-related demos, avoid leaking security anti-patterns (no credential specificity in generic errors).
