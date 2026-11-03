"""Tests focus on leakage, accounting, costs, and unfillable orders."""
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from ashare.backtest import is_rebalance, performance, run_backtest, trade_portfolio
from ashare.factors import FACTOR_NAMES, build_factors, factor_diagnostics, fit_weights, forward_labels
from ashare.data import load_data


@pytest.fixture
def bars():
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2023-01-02", periods=180)
    frames = []
    for j in range(15):
        code = f"{600000+j:06d}"
        close = 20 * np.cumprod(1 + rng.normal(.0002, .01, len(dates)))
        op = close * (1 + rng.normal(0, .002, len(dates)))
        frame = pd.DataFrame(dict(date=dates, code=code, open=op, close=close,
                                  high=np.maximum(op, close) * 1.01, low=np.minimum(op, close) * .99,
                                  adj_open=op, adj_close=close, volume=1e5,
                                  turnover=rng.uniform(.2, 5, len(dates)),
                                  pct_change=pd.Series(close).pct_change().fillna(0).to_numpy()*100))
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def config():
    return dict(top_n=3, rebalance="monthly", weighting="rolling_ic", train_days=30, test_days=21,
                label_horizon=5, min_history=65, min_ic_observations=5, initial_capital=1000,
                commission_bps=3, slippage_bps=5, sell_tax_bps=5,
                backtest_start="2023-06-01", backtest_end="2023-09-08", allow_unverified_data=True,
                benchmarks={"CSI300": "000300", "CSI500": "000905"})


def benches(bars):
    one = bars[bars.code == "600000"][["date", "open", "close"]]
    return pd.concat([one.assign(benchmark=c) for c in ["000300", "000905"]])


def evaluate(bars, cfg):
    factors = build_factors(bars, cfg["min_history"])
    labels = forward_labels(bars, cfg["label_horizon"])
    ic, _ = factor_diagnostics(factors, labels)
    return run_backtest(bars, benches(bars), factors, ic, cfg)


def test_factors_are_invariant_to_future_prices(bars):
    cutoff = sorted(bars.date.unique())[130]
    full = build_factors(bars, 65)
    prefix = build_factors(bars[bars.date <= cutoff], 65)
    assert_frame_equal(full.loc[:cutoff], prefix)


def test_labels_use_next_open_and_record_exit_date(bars):
    labels = forward_labels(bars, 5)
    one = bars[bars.code == "600000"].sort_values("date")
    signal = one.date.iloc[80]
    row = labels.loc[(signal, "600000")]
    assert row.forward_return == pytest.approx(one.adj_open.iloc[86] / one.adj_open.iloc[81] - 1)
    assert row.label_end == one.date.iloc[86]


def test_training_purges_unmatured_labels():
    rows = []
    for factor in FACTOR_NAMES:
        rows += [dict(date=pd.Timestamp("2023-01-02"), label_end=pd.Timestamp("2023-01-10"), factor=factor, rank_ic=.1),
                 dict(date=pd.Timestamp("2023-01-09"), label_end=pd.Timestamp("2023-02-10"), factor=factor, rank_ic=999)]
    ic = pd.DataFrame(rows)
    _, meta = fit_weights(ic, pd.Timestamp("2023-01-20"), pd.Timestamp("2023-01-01"), 1)
    assert meta["observations_min"] == 1
    assert meta["max_label_end"] == "2023-01-10"
    assert all(v == .1 for v in meta["mean_rank_ic"].values())


def test_ledger_buy_sell_costs_by_hand():
    c = config()
    c.update(commission_bps=10, slippage_bps=10, sell_tax_bps=10)
    units = pd.Series({"A": 0.})
    op = pd.Series({"A": 10.})
    gap, volume = pd.Series({"A": 0.}), pd.Series({"A": 100.})
    units, cash, trades = trade_portfolio(units, 1000., op, gap, volume, pd.Series({"A": 1.}), c)
    expected_notional = 1000 / 1.002
    assert units.A == pytest.approx(expected_notional / 10)
    assert cash == pytest.approx(0, abs=1e-8)
    assert trades[0]["cost"] == pytest.approx(expected_notional * .002)
    units, cash, trades = trade_portfolio(units, cash, op * 1.1, gap, volume, pd.Series(dtype=float), c)
    assert cash == pytest.approx(expected_notional * 1.1 * .997)
    assert units.A == 0


@pytest.mark.parametrize("gap,volume,side", [(.10, 100, "BUY"), (-.10, 100, "SELL"), (0, 0, "BUY"), (0, 0, "SELL")])
def test_limit_and_suspension_orders_do_not_fill(gap, volume, side):
    owned = side == "SELL"
    units = pd.Series({"A": 10. if owned else 0.})
    cash = 0. if owned else 100.
    u, balance, records = trade_portfolio(units, cash, pd.Series({"A": 10.}), pd.Series({"A": gap}),
                                          pd.Series({"A": volume}), pd.Series({"A": 0. if owned else 1.}), config())
    assert u.A == units.A and balance == cash
    assert records[0]["status"] == "BLOCKED"
    assert records[0]["notional"] == records[0]["cost"] == 0


def test_first_day_loss_is_in_drawdown():
    m = performance([-.1, 0., .02])
    assert m["max_drawdown"] == pytest.approx(-.1)
    assert m["total_return"] == pytest.approx(.9 * 1.02 - 1)


def test_month_and_week_schedule():
    assert is_rebalance(pd.Timestamp("2024-01-31"), pd.Timestamp("2024-02-01"), "monthly")
    assert not is_rebalance(pd.Timestamp("2024-02-01"), pd.Timestamp("2024-02-02"), "monthly")
    assert is_rebalance(pd.Timestamp("2024-02-02"), pd.Timestamp("2024-02-05"), "weekly")


def test_end_to_end_future_poisoning_does_not_change_past_trades(bars):
    cfg = config()
    cutoff = pd.Timestamp("2023-08-01")
    original = evaluate(bars, cfg)
    poisoned = bars.copy()
    future = poisoned.date > cutoff
    # Huge future return shock changes future labels and diagnostics, never prior decisions.
    poisoned.loc[future, ["open", "close", "adj_open", "adj_close", "high", "low"]] *= 3
    other = evaluate(poisoned, cfg)
    assert_frame_equal(original.daily[original.daily.date <= cutoff], other.daily[other.daily.date <= cutoff])
    assert_frame_equal(original.trades[original.trades.date <= cutoff], other.trades[other.trades.date <= cutoff])
    assert all(f["max_label_end"] <= f["train_cutoff"] < f["test_start"] for f in original.folds)
    assert (original.daily.cash >= 0).all()


def test_positive_costs_reduce_flat_price_wealth():
    units = pd.Series({"A": 0., "B": 0.})
    op = pd.Series({"A": 10., "B": 10.})
    gap, volume = op * 0, op * 100
    cash = 1000.
    for code in ["A", "B", "A", "B"]:
        units, cash, _ = trade_portfolio(units, cash, op, gap, volume, pd.Series({code: 1.}), config())
    assert 0 < cash + (units * op).sum() < 1000


def test_unknown_missing_quote_is_explicit_and_untradeable(bars, tmp_path):
    cfg = config()
    cfg["universe"] = {c: c for c in bars.code.unique()}
    missing_day = sorted(bars.date.unique())[80]
    sparse = bars[~((bars.code == "600000") & (bars.date == missing_day))]
    sparse.to_csv(tmp_path / "prices.csv", index=False)
    benches(bars).to_csv(tmp_path / "benchmarks.csv", index=False)
    aligned, _ = load_data(tmp_path, cfg)
    one = aligned[aligned.code == "600000"].set_index("date")
    assert not one.loc[missing_day, "observed"]
    assert one.loc[missing_day, "volume"] == 0
    assert one.loc[missing_day, "close"] == one.loc[:missing_day, "close"].iloc[-2]
    f = build_factors(aligned, 65).xs("600000", level="code")
    assert not f.loc[missing_day, "eligible"]
    assert not f.iloc[100].eligible
    assert f.iloc[141].eligible
    labels = forward_labels(aligned, 5)
    entry_signal = sorted(bars.date.unique())[79]
    assert pd.isna(labels.loc[(entry_signal, "600000"), "forward_return"])


def test_loader_rejects_missing_endpoint(bars, tmp_path):
    cfg = config()
    cfg["universe"] = {c: c for c in bars.code.unique()}
    end = bars.date.max()
    bars[~((bars.code == "600000") & (bars.date == end))].to_csv(tmp_path / "prices.csv", index=False)
    benches(bars).to_csv(tmp_path / "benchmarks.csv", index=False)
    with pytest.raises(ValueError, match="Missing endpoint"):
        load_data(tmp_path, cfg)
