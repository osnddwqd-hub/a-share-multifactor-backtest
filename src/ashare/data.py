"""Public daily bars: explicit downloads, immutable CSV snapshots, no credentials."""
from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

FIELDS = "date,open,close,high,low,volume,amount,amplitude,pct_change,change,turnover".split(",")


def _bars(code: str, market: int, start: str, end: str, adjust: int) -> pd.DataFrame:
    params = dict(secid=f"{market}.{code}", klt=101, fqt=adjust,
                  beg=start.replace("-", ""), end=end.replace("-", ""),
                  fields1="f1,f2,f3,f4,f5,f6",
                  fields2="f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61")
    url = "https://push2his.eastmoney.com/api/qt/stock/kline/get?" + urlencode(params)
    error = None
    for attempt in range(3):
        try:
            req = Request(url, headers={"User-Agent": "Mozilla/5.0", "Referer": "https://quote.eastmoney.com/"})
            with urlopen(req, timeout=30) as response:
                payload = json.load(response)
            data = payload.get("data")
            if not data or not data.get("klines"):
                raise ValueError(f"Empty market data: {code}, adjust={adjust}")
            frame = pd.DataFrame([s.split(",") for s in data["klines"]], columns=FIELDS)
            for col in FIELDS[1:]:
                frame[col] = pd.to_numeric(frame[col], errors="raise")
            frame["date"] = pd.to_datetime(frame["date"])
            return frame
        except Exception as exc:
            error = exc
            if attempt < 2:
                time.sleep(attempt + 1)
    raise RuntimeError(f"Download failed for {code}; no synthetic fallback") from error


def download(config: dict, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    cache = dest / "cache"
    cache.mkdir(exist_ok=True)
    start, end = config["data_start"], config["data_end"]

    def stock(code):
        # Cache keys include the full request interval and adjustment convention.
        path = cache / f"{code}_{start}_{end}.csv"
        if path.exists():
            return pd.read_csv(path, dtype={"code": str}, parse_dates=["date"])
        market = int(code.startswith("6"))
        raw = _bars(code, market, start, end, 0)
        adj = _bars(code, market, start, end, 2)  # Eastmoney: 2 = backward adjusted.
        adj = adj[["date", "open", "close"]].rename(columns={"open": "adj_open", "close": "adj_close"})
        frame = raw.merge(adj, on="date", how="outer", validate="one_to_one")
        if frame.isna().any().any():
            raise ValueError(f"Unaligned raw/adjusted bars: {code}")
        frame["code"] = code
        frame["name"] = config["universe"][code]
        frame.to_csv(path, index=False, float_format="%.8g")
        return frame

    frames = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = {pool.submit(stock, c): c for c in config["universe"]}
        for job in as_completed(jobs):
            frames.append(job.result())
            print(f"Downloaded {jobs[job]} ({len(frames)}/{len(jobs)})", flush=True)
    stocks = pd.concat(frames).sort_values(["date", "code"])
    benches = []
    for name, code in config["benchmarks"].items():
        frame = _bars(code, 1, start, end, 0)[["date", "open", "close"]]
        frame["benchmark"] = code
        frame["name"] = name
        benches.append(frame)
    stocks.to_csv(dest / "prices.csv", index=False, float_format="%.8g")
    pd.concat(benches).sort_values(["date", "benchmark"]).to_csv(dest / "benchmarks.csv", index=False)
    manifest = {
        "source": "Eastmoney public daily K-line endpoint",
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "requested_start": start, "requested_end": end,
        "adjustment": "fqt=2 backward-adjusted OHLC for returns; fqt=0 raw OHLC for trade gates",
        "universe_type": "30 manually fixed main-board stocks, NOT historical index membership; survivorship/selection bias",
        "synthetic": False,
        "files": {p.name: {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size}
                  for p in [dest / "prices.csv", dest / "benchmarks.csv"]},
    }
    (dest / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def load_data(directory: Path, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists() and not config.get("allow_unverified_data", False):
        raise ValueError("Missing manifest.json; set allow_unverified_data=true only for explicitly unverified custom data")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("synthetic") is not False:
            raise ValueError("Only real-data snapshots are accepted")
        for filename, meta in manifest["files"].items():
            if hashlib.sha256((directory / filename).read_bytes()).hexdigest() != meta["sha256"]:
                raise ValueError(f"Snapshot hash mismatch: {filename}")
    prices = pd.read_csv(directory / "prices.csv", dtype={"code": str}, parse_dates=["date"])
    benchmarks = pd.read_csv(directory / "benchmarks.csv", dtype={"benchmark": str}, parse_dates=["date"])
    required = {"date", "code", "open", "high", "low", "close", "adj_open", "adj_close", "volume", "turnover", "pct_change"}
    if required - set(prices):
        raise ValueError(f"Missing columns: {required - set(prices)}")
    if prices.duplicated(["date", "code"]).any() or benchmarks.duplicated(["date", "benchmark"]).any():
        raise ValueError("Duplicate bars")
    if set(config["universe"]) != set(prices.code):
        raise ValueError("Stock universe does not match snapshot")
    if set(config["benchmarks"].values()) != set(benchmarks.benchmark):
        raise ValueError("Benchmark universe does not match snapshot")
    for col in ["open", "high", "low", "close", "adj_open", "adj_close"]:
        if not np.isfinite(prices[col]).all() or (prices[col] <= 0).any():
            raise ValueError(f"Invalid price in {col}; missing data cannot silently become zero returns")
    if prices[["volume", "turnover", "pct_change"]].isna().any().any():
        raise ValueError("Missing volume, turnover or raw return")
    if not np.isfinite(prices[["volume", "turnover", "pct_change"]]).all().all():
        raise ValueError("Non-finite volume, turnover or raw return")
    if (prices[["volume", "turnover"]] < 0).any().any() or (prices["pct_change"] <= -100).any():
        raise ValueError("Negative volume/turnover or impossible price change")
    if not np.isfinite(benchmarks[["open", "close"]]).all().all() or (benchmarks[["open", "close"]] <= 0).any().any():
        raise ValueError("Invalid benchmark prices")
    calendar = pd.DatetimeIndex(sorted(benchmarks.date.unique()))
    for code, frame in benchmarks.groupby("benchmark"):
        if not pd.DatetimeIndex(sorted(frame.date)).equals(calendar):
            raise ValueError(f"Incomplete benchmark calendar: {code}")
    # Explicitly align interior missing quotes; never label unknown missing data as confirmed suspensions.
    # Carry prices only for valuation, block trading and factor eligibility on imputed dates.
    # Reject missing endpoints and long unknown gaps, which can hide listings/delistings or data failures.
    aligned = []
    for code, frame in prices.groupby("code"):
        frame = frame.set_index("date").sort_index()
        if frame.index[0] != calendar[0] or frame.index[-1] != calendar[-1]:
            raise ValueError(f"Missing endpoint for {code}; listing/delisting coverage is required")
        g = frame.reindex(calendar)
        observed = pd.Series(g.index.isin(frame.index), index=g.index)
        runs = (~observed).groupby(observed.cumsum()).sum()
        if runs.max() > config.get("max_missing_gap", 15):
            raise ValueError(f"Unknown quote gap exceeds allowed length: {code}")
        missing = ~observed
        g["observed"] = observed
        g["code"] = code
        g["name"] = config["universe"][code]
        g["close"] = g.close.ffill()
        g["adj_close"] = g.adj_close.ffill()
        for col in ["open", "high", "low"]:
            g.loc[missing, col] = g.loc[missing, "close"]
        g.loc[missing, "adj_open"] = g.loc[missing, "adj_close"]
        for col in ["volume", "turnover", "pct_change", "amount", "amplitude", "change"]:
            if col in g:
                g.loc[missing, col] = 0
        g.index.name = "date"
        aligned.append(g.reset_index())
    return pd.concat(aligned).sort_values(["date", "code"]), benchmarks.sort_values(["date", "benchmark"])
