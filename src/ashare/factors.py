"""Trailing-only signals and explicitly future-dated research labels."""
import numpy as np
import pandas as pd

FACTOR_NAMES = ["momentum_60_20", "reversal_5", "low_volatility_20", "low_turnover_20"]


def build_factors(prices: pd.DataFrame, min_history: int = 120) -> pd.DataFrame:
    frames = []
    for code, g in prices.groupby("code", sort=True):
        g = g.sort_values("date").copy()
        p = g.adj_close
        r = p.pct_change(fill_method=None)
        g["momentum_60_20"] = p.shift(20) / p.shift(60) - 1
        g["reversal_5"] = -(p / p.shift(5) - 1)
        g["low_volatility_20"] = -r.rolling(20, min_periods=20).std(ddof=0) * np.sqrt(252)
        g["low_turnover_20"] = -g.turnover.rolling(20, min_periods=20).mean()
        observed = g["observed"] if "observed" in g else pd.Series(True, index=g.index)
        # A quote gap must not create a spurious low-volatility / low-turnover signal.
        complete = observed.astype(int).rolling(61, min_periods=61).sum().eq(61)
        g["eligible"] = (np.arange(len(g)) + 1 >= min_history) & (g.volume > 0) & complete
        frames.append(g[["date", "code", "eligible"] + FACTOR_NAMES])
    factors = pd.concat(frames).set_index(["date", "code"]).sort_index()
    factors.loc[~factors.eligible, FACTOR_NAMES] = np.nan
    for name in FACTOR_NAMES:
        values = factors[name].unstack("code")
        lo, hi = values.quantile(.01, axis=1), values.quantile(.99, axis=1)
        clipped = values.clip(lower=lo, upper=hi, axis=0)
        sd = clipped.std(axis=1, ddof=0)
        standardized = clipped.sub(clipped.mean(axis=1), axis=0).div(sd.replace(0, np.nan), axis=0)
        # An all-identical valid cross section has no predictive ordering.
        standardized.loc[sd.eq(0)] = clipped.loc[sd.eq(0)].where(clipped.loc[sd.eq(0)].isna(), 0)
        factors[name] = standardized.stack(future_stack=True).reindex(factors.index)
    return factors


def forward_labels(prices: pd.DataFrame, horizon: int = 21) -> pd.DataFrame:
    """Signal at close t, entry open t+1, exit open t+1+h. Never a model feature."""
    op = prices.pivot(index="date", columns="code", values="adj_open").sort_index()
    ret = op.shift(-(horizon + 1)) / op.shift(-1) - 1
    volume = prices.pivot(index="date", columns="code", values="volume").sort_index()
    valid = volume.shift(-1).gt(0) & volume.shift(-(horizon + 1)).gt(0)
    if "observed" in prices:
        observed = prices.pivot(index="date", columns="code", values="observed").sort_index()
        valid &= observed.shift(-1).eq(True) & observed.shift(-(horizon + 1)).eq(True)
    ret = ret.where(valid)
    labels = ret.stack(future_stack=True).rename("forward_return").to_frame()
    dates = pd.Series(op.index, index=op.index).shift(-(horizon + 1))
    labels["label_end"] = labels.index.get_level_values("date").map(dates)
    return labels


def factor_diagnostics(factors: pd.DataFrame, labels: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    joint = factors.join(labels)
    rows, groups = [], []
    for day, cross in joint.groupby(level="date"):
        for name in FACTOR_NAMES:
            x = cross[[name, "forward_return", "label_end"]].dropna()
            if len(x) < 10 or x[name].nunique() < 2 or x.forward_return.nunique() < 2:
                continue
            pearson = x[name].corr(x.forward_return)
            ranked = x[name].rank(method="average").corr(x.forward_return.rank(method="average"))
            rows.append(dict(date=day, factor=name, ic=pearson, rank_ic=ranked,
                             label_end=x.label_end.iloc[0], n=len(x)))
            # Deterministic tie breaks by code; groups are descriptive, overlapping horizon returns.
            x = x.sort_index().sort_values(name, kind="stable")
            for q, indices in enumerate(np.array_split(np.arange(len(x)), 5), 1):
                groups.append(dict(date=day, factor=name, group=q,
                                   forward_return=x.iloc[indices].forward_return.mean(), n=len(indices)))
    return pd.DataFrame(rows, columns=["date", "factor", "ic", "rank_ic", "label_end", "n"]), pd.DataFrame(groups, columns=["date", "factor", "group", "forward_return", "n"])


def fit_weights(ic: pd.DataFrame, cutoff: pd.Timestamp, train_start: pd.Timestamp,
                min_observations: int = 40, method: str = "rolling_ic") -> tuple[dict, dict]:
    """Only labels fully observed by cutoff can affect the next OOS fold."""
    matured = ic[(ic.date >= train_start) & (ic.date <= cutoff) & (ic.label_end <= cutoff)]
    counts = matured.groupby("factor").size().reindex(FACTOR_NAMES, fill_value=0)
    means = matured.groupby("factor").rank_ic.mean().reindex(FACTOR_NAMES).fillna(0)
    if method == "equal" or counts.min() < min_observations:
        weights = pd.Series(1 / len(FACTOR_NAMES), index=FACTOR_NAMES)
    else:
        # Directions are predefined hypotheses; negative IC receives zero weight, never flipped ex post.
        positive = means.clip(lower=0)
        weights = positive / positive.sum() if positive.sum() > 0 else means * 0 + 1 / len(FACTOR_NAMES)
    meta = {"train_start": str(train_start.date()), "train_cutoff": str(cutoff.date()),
            "max_label_end": str(matured.label_end.max().date()) if len(matured) else None,
            "observations_min": int(counts.min()), "mean_rank_ic": means.to_dict()}
    return weights.to_dict(), meta
