from __future__ import annotations

import numpy as np
import pandas as pd


def _entropy_from_counts(counts: np.ndarray) -> float:
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log(probs)))


def _normalize_0_1(values: pd.Series) -> pd.Series:
    min_v = values.min()
    max_v = values.max()
    return (values - min_v) / (max_v - min_v + np.finfo(float).eps)


def _mutual_information_hist(x: np.ndarray, y: np.ndarray, bins: int = 20) -> tuple[float, float]:
    # Quantile-based bin edges are more stable when distributions are skewed.
    x_edges = np.unique(np.quantile(x, np.linspace(0, 1, bins + 1)))
    y_edges = np.unique(np.quantile(y, np.linspace(0, 1, bins + 1)))

    if len(x_edges) < 3 or len(y_edges) < 3:
        return 0.0, 0.0

    xy_counts = np.histogram2d(x, y, bins=(x_edges, y_edges))[0]
    x_counts = xy_counts.sum(axis=1)
    y_counts = xy_counts.sum(axis=0)

    total = xy_counts.sum()
    if total == 0:
        return 0.0, 0.0

    pxy = xy_counts / total
    px = x_counts / total
    py = y_counts / total

    eps = np.finfo(float).eps
    mi = np.sum(pxy * np.log((pxy + eps) / (px[:, None] * py[None, :] + eps)))

    hx = _entropy_from_counts(x_counts)
    hy = _entropy_from_counts(y_counts)
    nmi = mi / max(np.sqrt(hx * hy), eps)

    return float(mi), float(nmi)


def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    sse = float(np.sum((y_true - y_pred) ** 2))
    sst = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if sst <= np.finfo(float).eps:
        return 0.0
    return max(0.0, min(1.0, 1.0 - sse / sst))


def _linear_r2_single(x: np.ndarray, y: np.ndarray) -> float:
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    denom = float(np.sum((x - x_mean) ** 2))
    if denom <= np.finfo(float).eps:
        return 0.0

    beta = float(np.sum((x - x_mean) * (y - y_mean)) / denom)
    alpha = y_mean - beta * x_mean
    y_hat = alpha + beta * x
    return _safe_r2(y, y_hat)


def _linear_r2_multi(X: np.ndarray, y: np.ndarray) -> float:
    X_design = np.column_stack([np.ones(len(X)), X])
    try:
        coefs, *_ = np.linalg.lstsq(X_design, y, rcond=None)
    except np.linalg.LinAlgError:
        return 0.0

    y_hat = X_design @ coefs
    return _safe_r2(y, y_hat)


def intervention_priority(
    df: pd.DataFrame,
    candidate_vars: list[str] | None = None,
    w_corr: float = 0.10,
    w_nmi: float = 0.20,
    w_unique_outbound: float = 0.40,
    w_rootness: float = 0.30,
    w_var: float = 0.00,
    apply_rootness_penalty: bool = False,
    bins: int = 20,
) -> pd.DataFrame:
    numeric_df = df.select_dtypes(include=["number"]).copy()
    numeric_df = numeric_df.loc[:, ~numeric_df.columns.str.contains(r"^Unnamed")]

    if candidate_vars is None:
        candidate_vars = list(numeric_df.columns)

    if len(candidate_vars) < 2:
        raise ValueError("Need at least 2 numeric candidate variables for prioritization.")

    missing = [c for c in candidate_vars if c not in numeric_df.columns]
    if missing:
        raise ValueError(f"Missing candidate columns in dataframe: {missing}")

    data = numeric_df[candidate_vars]

    corr = data.corr(method="pearson").abs()
    corr_score = (corr.sum(axis=1) - 1.0) / (len(candidate_vars) - 1)

    nmi_mat = pd.DataFrame(0.0, index=candidate_vars, columns=candidate_vars)
    for i, col_i in enumerate(candidate_vars):
        for j, col_j in enumerate(candidate_vars):
            if j <= i:
                continue
            _, nmi = _mutual_information_hist(data[col_i].to_numpy(), data[col_j].to_numpy(), bins=bins)
            nmi_mat.loc[col_i, col_j] = nmi
            nmi_mat.loc[col_j, col_i] = nmi

    nmi_score = nmi_mat.sum(axis=1) / (len(candidate_vars) - 1)

    # Unique outbound effect proxy: average delta-R2 when adding source to predict each target.
    unique_outbound_rows = []
    for source in candidate_vars:
        delta_r2_values = []
        for target in candidate_vars:
            if target == source:
                continue
            base_predictors = [c for c in candidate_vars if c not in (source, target)]

            if base_predictors:
                r2_base = _linear_r2_multi(data[base_predictors].to_numpy(), data[target].to_numpy())
                r2_with_source = _linear_r2_multi(
                    data[base_predictors + [source]].to_numpy(), data[target].to_numpy()
                )
            else:
                r2_base = 0.0
                r2_with_source = _linear_r2_single(data[source].to_numpy(), data[target].to_numpy())

            delta_r2_values.append(max(0.0, r2_with_source - r2_base))

        unique_outbound_rows.append(float(np.mean(delta_r2_values)))
    unique_outbound_score = pd.Series(unique_outbound_rows, index=candidate_vars)

    # Inbound predictability proxy. High value can indicate "downstream-like" behavior,
    # but this is not causal proof without interventions.
    inbound_rows = []
    for target in candidate_vars:
        predictors = [c for c in candidate_vars if c != target]
        r2 = _linear_r2_multi(data[predictors].to_numpy(), data[target].to_numpy())
        inbound_rows.append(r2)
    inbound_score = pd.Series(inbound_rows, index=candidate_vars)

    var = data.var(ddof=1)
    var_norm = _normalize_0_1(var)

    corr_norm = _normalize_0_1(corr_score)
    nmi_norm = _normalize_0_1(nmi_score)
    unique_outbound_norm = _normalize_0_1(unique_outbound_score)
    rootness_score = 1.0 - _normalize_0_1(inbound_score)
    sink_likelihood = 1.0 - _normalize_0_1(0.6 * rootness_score + 0.4 * unique_outbound_norm)

    rootness_weight = w_rootness if apply_rootness_penalty else 0.0

    priority = (
        w_corr * corr_norm
        + w_nmi * nmi_norm
        + w_unique_outbound * unique_outbound_norm
        + rootness_weight * rootness_score
        + w_var * var_norm
    )

    result = pd.DataFrame(
        {
            "variable": candidate_vars,
            "corr_score": corr_norm.values,
            "nmi_score": nmi_norm.values,
            "unique_outbound_score": unique_outbound_norm.values,
            "rootness_score": rootness_score.values,
            "sink_likelihood": sink_likelihood.values,
            "raw_unique_outbound_delta_r2": unique_outbound_score.values,
            "inbound_r2_score": inbound_score.values,
            "variance_score": var_norm.values,
            "priority_score": priority.values,
        }
    ).sort_values("priority_score", ascending=False)

    return result


def main() -> None:
    df = pd.read_csv(r"data\data_1504.csv")

    # Set candidate_vars = None to use all numeric variables automatically.
    candidate_vars = None

    ranking = intervention_priority(
        df,
        candidate_vars=candidate_vars,
        apply_rootness_penalty=False,
    )

    print("Intervention priority ranking (higher is better first intervention):")
    print("Note: ranking is assumption-light by default (no sink penalty enabled).")
    print("Use apply_rootness_penalty=True only with domain evidence for downstream/sink structure.")
    print(ranking.to_string(index=False))


if __name__ == "__main__":
    main()
