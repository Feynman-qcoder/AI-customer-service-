"""``python -m evals.run_resume_benchmark`` entry point."""

from __future__ import annotations

import sys

from evals.benchmark.cli import main

if __name__ == "__main__":
    sys.exit(main())
