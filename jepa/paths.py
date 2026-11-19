"""Filesystem layout for the project.

Every path is derived from the package location, so commands resolve the same
``data/`` and ``runs/`` directories regardless of the current working
directory.
"""

from __future__ import annotations

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent

DATA_DIR = PROJECT_ROOT / "data"
TOKENIZER_DIR = DATA_DIR / "tokenizer_32k"
FINEWEB_DIR = DATA_DIR / "fineweb_edu_500m_32k"

RUNS_DIR = PROJECT_ROOT / "runs"
DEFAULT_RUN_DIR = RUNS_DIR / "jepa"
DEFAULT_DECODER_RUN_DIR = RUNS_DIR / "jepa-decoder"
