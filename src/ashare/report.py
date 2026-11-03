"""Reproducible static research outputs (CSV, JSON, SVG, Markdown, HTML)."""
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .backtest import performance
from .factors import FACTOR_NAMES

COLORS = ["#0f766e", "#5975a4", "#d28b36"]


def _save(fig, path):
    fig.savefig(path, bbox_inches="tight", facecolor="#ffffff", metadata={"Date": None})
    plt.close(fig)


def _pct(x):
    return "—" if x is None else f"{x:.2%}"


def write_report(result, ic, groups, config: dict, out: Path, manifest: dict):
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titleweight": "bold", "svg.fonttype": "none"})
    daily = result.daily.copy()
    metrics = {"strategy": performance(daily.strategy_return)}
    for name, code in config["benchmarks"].items():
        metrics[code] = {"name": name, **performance(daily[f"benchmark_return_{code}"])}
    metrics["execution"] = dict(total_cost=float(daily.cost.sum()),
                                gross_traded=float(daily.gross_traded.sum()),
                                average_daily_turnover=float(daily.turnover.mean()),
                                filled_orders=int((result.trades.notional > 0).sum()),
                                blocked_orders=int((result.trades.status == "BLOCKED").sum()))
    for code in config["benchmarks"].values():
        a = daily.strategy_return - daily[f"benchmark_return_{code}"]
        te = a.std(ddof=0) * np.sqrt(252)
        metrics["strategy"][f"information_ratio_vs_{code}"] = float(a.mean() * 252 / te) if te else None
        metrics["strategy"][f"relative_wealth_return_vs_{code}"] = float(
            (1 + daily.strategy_return).prod() / (1 + daily[f"benchmark_return_{code}"]).prod() - 1)
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    (out / "folds.json").write_text(json.dumps(result.folds, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    (out / "run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "data_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    daily.to_csv(out / "daily_nav.csv", index=False, float_format="%.10g")
    result.trades.to_csv(out / "trades.csv", index=False, float_format="%.10g")
    result.holdings.to_csv(out / "holdings.csv", index=False, float_format="%.10g")
    result.selections.to_csv(out / "selections.csv", index=False, float_format="%.10g")
    ic.to_csv(out / "factor_ic_all.csv", index=False, float_format="%.10g")
    # Report test-period diagnostics separately; never use this table for past weights.
    oos_ic = ic[(ic.date >= config["backtest_start"]) & (ic.label_end <= pd.Timestamp(config["backtest_end"]))]
    summary = oos_ic.groupby("factor").agg(mean_ic=("ic", "mean"), mean_rank_ic=("rank_ic", "mean"),
                                              rank_ic_std=("rank_ic", "std"), count=("rank_ic", "size"))
    summary["positive_ratio"] = oos_ic.assign(positive=oos_ic.rank_ic > 0).groupby("factor").positive.mean()
    # Horizon labels overlap: do not present naive annualized ICIR or t-statistics.
    summary["icir_unannualized"] = summary.mean_rank_ic / summary.rank_ic_std.replace(0, np.nan)
    summary.to_csv(out / "factor_summary_oos.csv", float_format="%.10g")
    valid_days = set(oos_ic.date)
    group_mean = groups[groups.date.isin(valid_days)].groupby(["factor", "group"]).forward_return.mean().unstack("group")
    group_mean.to_csv(out / "factor_groups_oos.csv", float_format="%.10g")
    fold_rows = []
    for f, g in daily.groupby("fold"):
        fold_rows.append({"fold": int(f), "start": str(g.date.iloc[0].date()), "end": str(g.date.iloc[-1].date()),
                          **performance(g.strategy_return), "cost": float(g.cost.sum())})
    pd.DataFrame(fold_rows).to_csv(out / "rolling_windows.csv", index=False)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7.2), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    curves = {"Multifactor (net)": (1 + daily.strategy_return).cumprod(),
              "CSI 300 (price index)": daily.benchmark_000300,
              "CSI 500 (price index)": daily.benchmark_000905}
    # Show initial capital at the last training day's close for a truthful origin.
    baseline_date = pd.Timestamp(result.folds[0]["train_cutoff"])
    x = pd.concat([pd.Series([baseline_date]), daily.date], ignore_index=True)
    for color, (label, values) in zip(COLORS, curves.items()):
        wealth = pd.concat([pd.Series([1.]), values], ignore_index=True)
        axes[0].plot(x, wealth, label=label, color=color, linewidth=1.8)
        axes[1].plot(x, (wealth / wealth.cummax() - 1) * 100, color=color, linewidth=1.3)
    axes[0].set_title("A-SHARE MULTIFACTOR | WALK-FORWARD RESEARCH", loc="left", pad=16)
    axes[0].set_ylabel("Normalized wealth")
    axes[0].legend(loc="upper left", frameon=False, ncol=1)
    axes[1].set_ylabel("Drawdown (%)")
    for ax in axes:
        ax.grid(alpha=.15)
    fig.text(.12, .005, "30 fixed stocks | next-open execution | costs included | not a historical index universe", color="#667085", fontsize=9)
    fig.tight_layout(pad=1.8)
    _save(fig, out / "performance.svg")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    names = ["Momentum 60-20", "5-day reversal", "Low volatility", "Low turnover"]
    vals = summary.reindex(FACTOR_NAMES).mean_rank_ic
    axes[0].barh(names[::-1], vals.iloc[::-1], color=[COLORS[0] if v >= 0 else "#b66158" for v in vals.iloc[::-1]])
    axes[0].axvline(0, color="#98a2b3", linewidth=.8)
    axes[0].set_title("OOS mean Rank IC (21-day labels)", loc="left")
    for factor, label, color in zip(FACTOR_NAMES, names, [*COLORS, "#8b6bb1"]):
        if factor in group_mean.index:
            axes[1].plot(group_mean.columns, group_mean.loc[factor] * 100, marker="o", label=label, color=color)
    axes[1].set_title("OOS average forward group return", loc="left")
    axes[1].set_xlabel("Quintile (1 = low factor, 5 = high factor)")
    axes[1].set_ylabel("21-day return (%)")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].set_xticks(range(1, 6))
    fig.tight_layout()
    _save(fig, out / "factors.svg")

    fig, ax = plt.subplots(figsize=(12, 3.6))
    wf = pd.DataFrame([f["weights"] for f in result.folds], index=[f["test_start"] for f in result.folds])
    bottom = np.zeros(len(wf))
    for factor, label, color in zip(FACTOR_NAMES, names, [*COLORS, "#8b6bb1"]):
        ax.bar(range(len(wf)), wf[factor], bottom=bottom, label=label, color=color)
        bottom += wf[factor].to_numpy()
    ax.set_xticks(range(len(wf)), wf.index, rotation=30, ha="right")
    ax.set_ylim(0, 1.02)
    ax.set_title("Frozen factor weights per out-of-sample fold", loc="left")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(.5, 1.27), frameon=False, fontsize=9)
    fig.tight_layout()
    _save(fig, out / "weights.svg")

    rows = []
    for key, label in [("strategy", "多因子策略（扣费）"), ("000300", "沪深300价格指数"), ("000905", "中证500价格指数")]:
        m = metrics[key]
        rows.append(f"|{label}|{_pct(m['total_return'])}|{_pct(m['annualized_return'])}|{_pct(m['max_drawdown'])}|{m['sharpe']:.2f}|{m['calmar']:.2f}|")
    table = "|组合|累计收益|年化收益|最大回撤|Sharpe|Calmar|\n|---|---:|---:|---:|---:|---:|\n" + "\n".join(rows)
    text = f"""# 第一版真实行情回测报告

回测期间：{daily.date.iloc[0].date()} — {daily.date.iloc[-1].date()}；{len(daily)} 个交易日，{len(result.folds)} 个连续样本外窗口。

{table}

![净值与回撤](performance.svg)

初始资金 {config['initial_capital']:,.0f} 元；总交易成本 {daily.cost.sum():,.2f} 元。
每侧手续费 {config['commission_bps']} bp、滑点 {config['slippage_bps']} bp，卖出附加税费 {config['sell_tax_bps']} bp（固定研究假设）。

![因子检验](factors.svg)

分组收益是重叠的未来21个交易日平均收益，不是可交易分组净值；ICIR未年化，不做独立样本显著性宣称。

![滚动因子权重](weights.svg)

训练使用过去 {config['train_days']} 个交易日，剔除标签尚未成熟的日期；每 {config['test_days']} 个交易日重新估计因子权重，测试窗口内冻结。月初及新窗口首日用前一交易日收盘信号调仓。

股票池固定为30只主板股票，存在存活与人为选择偏差。后复权收益与价格指数的分红口径不同。采用分数份额和开盘价成交近似，未模拟整手、最低佣金、盘口队列、历史ST名单和完整退市事件。上述收益仅用于检验研究流程。

原始数据及哈希见 `../../data/manifest.json`；逐笔成交、每日持仓、选股打分、窗口结果分别见本目录 CSV。当前截图未按收益筛选参数，保留不利结果。
"""
    (out / "REPORT.md").write_text(text, encoding="utf-8")
    svg = [(out / name).read_text(encoding="utf-8").split("<svg", 1)[1] for name in ["performance.svg", "factors.svg", "weights.svg"]]
    body = "".join("<section><svg" + x + "</section>" for x in svg)
    s = metrics["strategy"]
    cards = "".join(f"<div><small>{label}</small><strong>{value}</strong></div>" for label, value in [
        ("年化收益（扣费）", _pct(s["annualized_return"])), ("最大回撤", _pct(s["max_drawdown"])),
        ("Sharpe（无风险利率=0）", f"{s['sharpe']:.2f}"), ("连续样本外窗口", str(len(result.folds)))])
    doc = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>A股多因子选股 · 第一版研究报告</title><style>
body{{margin:0;background:#edf2f5;color:#172b3a;font:16px/1.7 system-ui,sans-serif}}main{{max-width:1120px;margin:48px auto;padding:0 24px}}h1{{font-size:34px;line-height:1.3}}.eyebrow{{color:#0f766e;letter-spacing:3px;font-size:12px}}.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin:28px 0}}.cards div,section{{background:white;border:1px solid #dce5eb;border-radius:12px;padding:20px}}small{{display:block;color:#667085}}strong{{display:block;font-size:28px}}section{{margin:20px 0;overflow:auto}}svg{{width:100%;height:auto}}.note{{border-left:4px solid #d28b36;padding:12px 18px;background:#fff8ec}}@media(max-width:700px){{.cards{{grid-template-columns:repeat(2,1fr)}}h1{{font-size:26px}}}}
</style><main><span class="eyebrow">QUANT RESEARCH / V1</span><h1>基于多因子模型的A股量化选股<br>与滚动回测系统</h1><p>{daily.date.iloc[0].date()} — {daily.date.iloc[-1].date()} · 30只固定主板股票 · 真实历史日线</p><div class="cards">{cards}</div>{body}<p class="note">固定股票池存在样本选择与存活偏差；后复权收益与价格指数分红口径不同；成交为分数份额近似。结果展示研究流程，不是实盘业绩。</p><p>数据来源：{html.escape(manifest.get('source', 'external CSV'))}。全部参数、成交、持仓与训练截止日期保存在同目录中。详细方法见仓库 docs/METHODOLOGY.md。</p></main></html>"""
    (out / "report.html").write_text(doc, encoding="utf-8")
    return metrics, table
