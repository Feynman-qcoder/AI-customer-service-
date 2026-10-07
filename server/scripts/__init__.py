"""Operational scripts for local development, seeding and evaluation.

Making this a regular package (rather than relying on implicit namespace
packages) keeps ``python -m scripts.bootstrap_local`` and ``mypy scripts``
resolving the same way regardless of how the interpreter is invoked.
"""
