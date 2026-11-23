"""Repository paths. Environment variables are read at their point of use, never here."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "config"
RUNS_DIR = REPO_ROOT / "runs"
OUT_DIR = REPO_ROOT / "out"
MODELS_TOML = CONFIG_DIR / "models.toml"
ENGAGEMENTS_DIR = REPO_ROOT / "data" / "engagements"
DEMO_CACHE_DIR = RUNS_DIR / "demo" / "cache"
