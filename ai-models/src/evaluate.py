"""Compatibility entry point for current evaluation scripts in root src/."""

from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[2]
runpy.run_path(str(ROOT / "src" / "evaluate_models.py"))
runpy.run_path(str(ROOT / "src" / "confusion_matrix.py"))