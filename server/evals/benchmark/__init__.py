"""RESUME_AGENT_BENCHMARK_V1 — reproducible benchmark package.

Contract highlights:
- Holdouts are frozen once (SHA-256 manifest) and never edited afterwards.
- Mock LLM/embedding results may only validate the engineering chain; real
  semantic scores require real providers and are reported as NOT_RUN otherwise.
- Evidence (results, manifest, sanitized logs) is written outside the repo to
  ``BENCHMARK_EVIDENCE_DIR``.
"""
