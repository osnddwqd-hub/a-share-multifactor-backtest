import argparse
import json
from pathlib import Path

from .backtest import run_backtest
from .data import download, load_data
from .factors import build_factors, factor_diagnostics, forward_labels
from .report import write_report


def main():
    parser = argparse.ArgumentParser(description="A-share multifactor V1 — download or offline backtest")
    parser.add_argument("command", choices=["download", "backtest"])
    parser.add_argument("--config", type=Path, default=Path("configs/v1.json"))
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("reports/local"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.command == "download":
        download(config, args.data)
        return
    prices, benches = load_data(args.data, config)
    args.output.mkdir(parents=True, exist_ok=True)
    imputed = prices[~prices.observed][["date", "code", "name"]].copy()
    imputed["handling"] = "unknown_missing_quote: carry prior close for valuation; no fill; factor excluded for 61 bars"
    imputed.to_csv(args.output / "missing_quotes.csv", index=False)
    factors = build_factors(prices, config["min_history"])
    labels = forward_labels(prices, config["label_horizon"])
    ic, groups = factor_diagnostics(factors, labels)
    result = run_backtest(prices, benches, factors, ic, config)
    manifest_path = args.data / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {"source": "user-supplied CSV; provenance unverified"}
    metrics, _ = write_report(result, ic, groups, config, args.output, manifest)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Report saved: {args.output / 'REPORT.md'}")


if __name__ == "__main__":
    main()
