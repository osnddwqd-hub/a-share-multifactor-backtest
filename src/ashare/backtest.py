"""Self-financing, fractional adjusted-unit research ledger with next-open execution."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .factors import FACTOR_NAMES, fit_weights


@dataclass
class BacktestResult:
    daily: pd.DataFrame
    trades: pd.DataFrame
    holdings: pd.DataFrame
    folds: list
    selections: pd.DataFrame


def is_rebalance(previous: pd.Timestamp, current: pd.Timestamp, frequency: str) -> bool:
    rule = {"monthly": "M", "weekly": "W-FRI"}[frequency]
    return previous.to_period(rule) != current.to_period(rule)


def trade_portfolio(units, cash, opens, raw_returns, volumes, target, config):
    """Sell first, then scale buys to available cash. Blocked orders are logged, not filled."""
    units = units.copy()
    nav = cash + float((units * opens).sum())
    target_value = target.reindex(units.index, fill_value=0) * nav
    current = units * opens
    buy_rate = (config["commission_bps"] + config["slippage_bps"]) / 10000
    sell_rate = buy_rate + config["sell_tax_bps"] / 10000
    records = []
    # Conservative main-board gate: prevent buying at >=9.5% raw open gap,
    # selling at <=-9.5%. Not a tick-level limit price/order-queue simulation.
    for code in units.index:
        wanted = max(0., current[code] - target_value[code])
        if wanted < 1e-8:
            continue
        reason = "suspended" if volumes[code] <= 0 else "down_limit_proxy" if raw_returns[code] <= -.095 else ""
        sold = 0. if reason else wanted
        cost = sold * sell_rate
        units[code] -= sold / opens[code]
        cash += sold - cost
        records.append(dict(code=code, side="SELL", requested=wanted, notional=sold, cost=cost,
                            price=opens[code], status="BLOCKED" if reason else "FILLED", reason=reason))
    desired = (target_value - units * opens).clip(lower=0)
    blocked = (volumes <= 0) | (raw_returns >= .095)
    allowed = desired.where(~blocked, 0)
    need = allowed.sum() * (1 + buy_rate)
    scale = min(1., max(0., cash) / need) if need > 0 else 0.
    for code in units.index:
        if desired[code] < 1e-8:
            continue
        bought = allowed[code] * scale
        cost = bought * buy_rate
        units[code] += bought / opens[code]
        cash -= bought + cost
        reason = "suspended" if volumes[code] <= 0 else "up_limit_proxy" if raw_returns[code] >= .095 else ""
        status = "BLOCKED" if reason else "FILLED" if scale >= 1 - 1e-12 else "PARTIAL_CASH"
        records.append(dict(code=code, side="BUY", requested=float(desired[code]), notional=bought, cost=cost,
                            price=opens[code], status=status, reason=reason))
    if cash < -1e-6 or (units < -1e-10).any():
        raise AssertionError("Self-financing long-only ledger violated")
    return units.clip(lower=0), max(0., cash), records


def run_backtest(prices, benchmarks, factors, ic, config) -> BacktestResult:
    if config["weighting"] not in {"equal", "rolling_ic"}:
        raise ValueError("weighting must be equal or rolling_ic")
    if config["rebalance"] not in {"monthly", "weekly"}:
        raise ValueError("rebalance must be monthly or weekly")
    for key in ["train_days", "test_days", "top_n", "label_horizon", "initial_capital"]:
        if config[key] <= 0:
            raise ValueError(f"{key} must be positive")
    if any(config[k] < 0 for k in ["commission_bps", "slippage_bps", "sell_tax_bps"]):
        raise ValueError("Costs cannot be negative")
    if config["top_n"] > prices.code.nunique():
        raise ValueError("top_n exceeds stock universe")
    wide = {col: prices.pivot(index="date", columns="code", values=col).sort_index()
            for col in ["adj_open", "adj_close", "open", "close", "pct_change", "volume"]}
    dates, codes = wide["adj_close"].index, wide["adj_close"].columns
    indices = np.flatnonzero((dates >= config["backtest_start"]) & (dates <= config["backtest_end"]))
    if len(indices) == 0 or indices[0] < config["train_days"] + config["min_history"]:
        raise ValueError("Insufficient history for trailing factors and first training window")
    # Raw pct_change is relative to the exchange's ex-rights reference close.
    reference = wide["close"] / (1 + wide["pct_change"] / 100)
    gaps = wide["open"] / reference - 1
    units = pd.Series(0., index=codes)
    cash = previous_nav = float(config["initial_capital"])
    daily, trades, holdings, folds, selections = [], [], [], [], []
    weights = {name: .25 for name in FACTOR_NAMES}
    bench_nav = {code: 1. for code in config["benchmarks"].values()}
    bopen = benchmarks.pivot(index="date", columns="benchmark", values="open")
    bclose = benchmarks.pivot(index="date", columns="benchmark", values="close")
    for offset, i in enumerate(indices):
        day, signal_day = dates[i], dates[i - 1]
        fold_start = offset % config["test_days"] == 0
        if fold_start:
            start = dates[i - config["train_days"]]
            weights, meta = fit_weights(ic, signal_day, start, config["min_ic_observations"], config["weighting"])
            last_i = indices[min(offset + config["test_days"] - 1, len(indices) - 1)]
            folds.append({"fold": len(folds) + 1, **meta, "test_start": str(day.date()),
                          "test_end": str(dates[last_i].date()), "weights": weights.copy()})
        records = []
        if fold_start or is_rebalance(signal_day, day, config["rebalance"]):
            cross = factors.xs(signal_day, level="date").dropna(subset=FACTOR_NAMES).copy()
            cross = cross[cross.eligible]
            cross["score"] = sum(cross[name] * weights[name] for name in FACTOR_NAMES)
            ranked = cross.sort_index().sort_values("score", ascending=False, kind="stable")
            chosen = ranked.head(config["top_n"])
            # If the eligible pool is smaller than top_n, preserve unused allocation as cash.
            target = pd.Series(1 / config["top_n"], index=chosen.index)
            units, cash, records = trade_portfolio(units, cash, wide["adj_open"].loc[day],
                                                  gaps.loc[day], wide["volume"].loc[day], target, config)
            # Execution-day zero volume is a realized non-fill gate, not a selection feature.
            # Explicit zero-volume and padded unknown-quote dates both prevent fills.
            for r in records:
                r.update(date=day, signal_date=signal_day, fold=len(folds))
                trades.append(r)
            for code, r in ranked.iterrows():
                selections.append(dict(signal_date=signal_day, execution_date=day, code=code,
                                       score=r.score, selected=code in chosen.index, fold=len(folds)))
        close_values = units * wide["adj_close"].loc[day]
        nav = cash + float(close_values.sum())
        for code in codes[units > 1e-12]:
            holdings.append(dict(date=day, code=code, adjusted_units=units[code],
                                 market_value=close_values[code], weight=close_values[code] / nav))
        row = dict(date=day, nav=nav, strategy_return=nav / previous_nav - 1, cash=cash,
                   cost=sum(t["cost"] for t in records), gross_traded=sum(t["notional"] for t in records),
                   turnover=sum(t["notional"] for t in records) / previous_nav,
                   holding_count=int((units > 1e-12).sum()), fold=len(folds))
        for code in bench_nav:
            base = bopen.loc[day, code] if offset == 0 else bclose.loc[dates[i - 1], code]
            change = bclose.loc[day, code] / base - 1
            bench_nav[code] *= 1 + change
            row[f"benchmark_{code}"] = bench_nav[code]
            row[f"benchmark_return_{code}"] = change
        daily.append(row)
        previous_nav = nav
    return BacktestResult(pd.DataFrame(daily), pd.DataFrame(trades), pd.DataFrame(holdings), folds, pd.DataFrame(selections))


def performance(returns, risk_free: float = 0.) -> dict:
    r = pd.Series(returns, dtype=float)
    if len(r) == 0 or not np.isfinite(r).all() or (r <= -1).any():
        raise ValueError("Invalid return series")
    nav = (1 + r).cumprod()
    # Include initial wealth 1, so a first-day loss counts in max drawdown.
    wealth = pd.concat([pd.Series([1.]), nav.reset_index(drop=True)], ignore_index=True)
    drawdown = wealth / wealth.cummax() - 1
    ann = nav.iloc[-1] ** (252 / len(r)) - 1
    vol = r.std(ddof=0) * np.sqrt(252)
    mdd = float(drawdown.min())
    return dict(total_return=float(nav.iloc[-1] - 1), annualized_return=float(ann),
                annualized_volatility=float(vol), max_drawdown=mdd,
                sharpe=float((r.mean() * 252 - risk_free) / vol) if vol > 1e-12 else None,
                calmar=float(ann / abs(mdd)) if mdd < -1e-12 else None,
                observations=len(r))
