"""Run from the repository root: python scripts/run_demo.py."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ashare.cli import main

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "backtest", "--output", "reports/local"]
    main()
