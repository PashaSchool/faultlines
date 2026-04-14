"""Symbol-level attribution for features and flows.

Extends the feature map with per-symbol attribution so AI agents
can request only the functions relevant to a flow instead of
reading entire files.

Rules:
  - Types, interfaces, and enums attribute to the feature level only
    (shared context, not flow-specific).
  - Functions, classes, and constants can attribute to multiple flows
    if they're used across user journeys.
  - File paths are always preserved as a fallback — AI agents can
    read the full file when symbol scope is too narrow.

Fully isolated module. Runs after the main pipeline when --symbols
is enabled in the CLI. Zero effect on the default analyze flow.
"""
