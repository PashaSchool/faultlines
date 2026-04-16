"""Impact analysis module.

Predicts which files, features, and flows will be affected by a set of
changes. Uses git co-change patterns (behavioral) instead of AST imports
(structural) — catches runtime, cross-boundary, and API-level dependencies
that static analysis misses.

Fully isolated from the analyzer and llm modules. Reads the feature-map
JSON as input and git history via subprocess. No LLM calls required.
"""
