from pathlib import Path
import re
import numpy as np
from scipy.stats import gaussian_kde
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import ttest_ind
from scipy.stats import ks_2samp
from scipy.stats import levene
from itertools import combinations

# =============================================================================
# ORIGINAL CODE
# =============================================================================

def get_csv_file_list(data_dir="./data"):
    data_path = Path(data_dir)
    if not data_path.exists():
        return []
    return [str(path) for path in data_path.glob("*.csv")]

def get_csv_file(csv_file_name, data_dir="./data"):
    tempList = get_csv_file_list(data_dir=data_dir)
    for file in tempList:
        if csv_file_name in file:
            return file
    return None

def load_csv_data(csv_file_name, data_dir="./data"):
    csv_file_path = get_csv_file(csv_file_name, data_dir=data_dir)
    if csv_file_path:
        return pd.read_csv(csv_file_path)
    return None

def get_columns(dataframe):
    return dataframe.columns.tolist()


def parse_intervention_target(file_name, columns=None):
    """
    Parse filenames like '-2a.csv' or '2d.csv' and return the target column.

    The leading signed number is treated as the intervention strength;
    the trailing letters identify the intervened column.
    """
    stem = Path(file_name).stem
    match = re.match(r"^[+-]?\d+([A-Za-z]+)$", stem)
    if not match:
        return None

    target = match.group(1)
    if columns is None:
        return target.upper()

    lookup = {str(column).lower(): column for column in columns}
    return lookup.get(target.lower(), target.upper())


def discover_intervention_files(intervention_dirs=("./data", "./testdata"),
                               exclude_files=None):
    """
    Find intervention CSVs across one or more folders.

    Files are only kept when their stem matches the expected pattern
    (for example '-2a' or '2d').
    """
    exclude_names = {
        Path(name).name.lower()
        for name in (exclude_files or [])
    }
    files = []

    for folder in intervention_dirs:
        folder_path = Path(folder)
        if not folder_path.exists():
            continue

        for file_path in sorted(folder_path.glob("*.csv")):
            if file_path.name.lower() in exclude_names:
                continue
            if parse_intervention_target(file_path.name) is None:
                continue
            files.append(file_path)

    return files


# MI Estimators

def kde_MI(x, y, grid_points=30):
    x, y = np.asarray(x), np.asarray(y)
    values = np.vstack([x, y])
    kernel = gaussian_kde(values)

    x_grid = np.linspace(x.min(), x.max(), grid_points)
    y_grid = np.linspace(y.min(), y.max(), grid_points)
    X, Y = np.meshgrid(x_grid, y_grid)
    positions = np.vstack([X.ravel(), Y.ravel()])

    p_xy = kernel(positions).reshape(grid_points, grid_points)
    p_xy /= np.sum(p_xy)

    p_x = np.sum(p_xy, axis=0)
    p_y = np.sum(p_xy, axis=1)
    p_x_p_y = p_x[np.newaxis, :] * p_y[:, np.newaxis]

    mask = p_xy > 1e-12
    return np.sum(p_xy[mask] * np.log(p_xy[mask] / p_x_p_y[mask]))


def binned_MI(x, y, bins=10):
    x, y = np.asarray(x), np.asarray(y)

    x_edges = np.quantile(x, np.linspace(0, 1, bins + 1))
    y_edges = np.quantile(y, np.linspace(0, 1, bins + 1))

    c_xy, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])
    p_xy = c_xy / np.sum(c_xy)

    p_x = np.sum(p_xy, axis=1)
    p_y = np.sum(p_xy, axis=0)
    p_x_p_y = p_x[:, np.newaxis] * p_y[np.newaxis, :]

    mask = p_xy > 0
    return np.sum(p_xy[mask] * np.log(p_xy[mask] / p_x_p_y[mask]))


def histogram_MI(x, y, n_bins=21):
    x, y = np.asarray(x), np.asarray(y)

    x_bins = np.linspace(x.min(), x.max(), n_bins)
    y_bins = np.linspace(y.min(), y.max(), n_bins)

    x_marginal = np.histogram(x, bins=x_bins)[0].astype(float)
    x_marginal /= x_marginal.sum()

    y_marginal = np.histogram(y, bins=y_bins)[0].astype(float)
    y_marginal /= y_marginal.sum()

    xy_joint, _, _ = np.histogram2d(x, y, bins=[x_bins, y_bins])
    xy_joint = xy_joint.astype(float)
    xy_joint /= xy_joint.sum()

    p_x_p_y = x_marginal[:, None] * y_marginal[None, :]
    mask = (xy_joint > 0) & (p_x_p_y > 0)
    return np.sum(xy_joint[mask] * np.log(xy_joint[mask] / p_x_p_y[mask]))


# method registry

MI_METHODS = {
    "kde":       kde_MI,
    "binned":    binned_MI,
    "histogram": histogram_MI,
}


# plotting

def plot_mutual_information(dataframe, method="kde"):
    """
    Plot pairwise MI matrix with KDE contours.

    Parameters
    ----------
    dataframe : pd.DataFrame
    method    : str — one of "kde", "binned", "histogram"
    """
    if method not in MI_METHODS:
        raise ValueError(f"Unknown method '{method}'. Choose from: {list(MI_METHODS)}")

    mi_fn = MI_METHODS[method]

    df   = dataframe.iloc[:, 1:]
    cols = get_columns(df)
    n    = len(cols)

    fig, axes = plt.subplots(n, n, figsize=(3.5 * n, 3.5 * n))
    axes = np.atleast_2d(axes)
    fig.suptitle(f"Mutual Information  —  method: {method}", fontsize=12, y=1.01)

    for i in range(n):
        for j in range(n):
            ax = axes[i, j]

            x = pd.to_numeric(df[cols[j]], errors="coerce").dropna().to_numpy()
            y = pd.to_numeric(df[cols[i]], errors="coerce").dropna().to_numpy()

            if len(x) == 0 or len(y) == 0 or np.std(x) == 0 or np.std(y) == 0:
                ax.axis("off")
                continue

            divider  = make_axes_locatable(ax)
            ax_top   = divider.append_axes("top",   size="22%", pad=0.05, sharex=ax)
            ax_right = divider.append_axes("right", size="22%", pad=0.05, sharey=ax)

            if i == j:
                ax.plot(x, x, color="steelblue", linewidth=1.2)
            else:
                mi = mi_fn(x, y)

                xx, yy    = np.mgrid[x.min():x.max():100j, y.min():y.max():100j]
                positions = np.vstack([xx.ravel(), yy.ravel()])
                kernel    = gaussian_kde(np.vstack([x, y]))
                f         = np.reshape(kernel(positions).T, xx.shape)
                ax.contour(xx, yy, f, colors="steelblue", linewidths=1.2, alpha=0.8)

                ax.text(
                    0.05, 0.95,
                    f"MI ({method}): {mi:.3f}",
                    transform=ax.transAxes,
                    fontsize=8,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
                )

            # Marginal density plots
            x_kde   = gaussian_kde(x)
            y_kde   = gaussian_kde(y)
            x_range = np.linspace(x.min(), x.max(), 100)
            y_range = np.linspace(y.min(), y.max(), 100)

            ax_top.plot(x_range,          x_kde(x_range), color="steelblue", linewidth=1.5)
            ax_right.plot(y_kde(y_range), y_range,         color="steelblue", linewidth=1.5)
            ax_top.axis("off")
            ax_right.axis("off")

            ax.set_xlabel(cols[j], fontsize=8)
            ax.set_ylabel(cols[i], fontsize=8)
            ax.tick_params(labelsize=6)

    plt.tight_layout()
    plt.show()

def get_percentiles(dataframe, column_name):
    if column_name not in dataframe.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame.")
    return dataframe[column_name].quantile([0.25, 0.75]).to_dict()

def get_conditional_dataframe(dataframe, column_name, lower_percentile=None, upper_percentile=None):
    if lower_percentile is not None and upper_percentile is not None:
        raise ValueError("Only one of lower_percentile or upper_percentile should be provided.")
    if column_name not in dataframe.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame.")
    if lower_percentile is not None:
        threshold = dataframe[column_name].quantile(lower_percentile)
        return dataframe[dataframe[column_name] < threshold]
    if upper_percentile is not None:
        threshold = dataframe[column_name].quantile(upper_percentile)
        return dataframe[dataframe[column_name] > threshold]

def mean_test(dataframe_1, dataframe_2, column_name):
    common_columns = set(dataframe_1.columns).intersection(set(dataframe_2.columns))
    if not common_columns:
        raise ValueError("No common columns found between the two DataFrames.")
    results = {}
    for column in common_columns:
        if pd.api.types.is_numeric_dtype(dataframe_1[column]) and pd.api.types.is_numeric_dtype(dataframe_2[column]):
            stat, p_value = ttest_ind(dataframe_1[column].dropna(), dataframe_2[column].dropna())
            results[column] = {"t_statistic": stat, "p_value": p_value}
    return results

def variance_test(dataframe_1, dataframe_2, column_name):
    common_columns = set(dataframe_1.columns).intersection(set(dataframe_2.columns))
    if not common_columns:
        raise ValueError("No common columns found between the two DataFrames.")
    results = {}
    for column in common_columns:
        if pd.api.types.is_numeric_dtype(dataframe_1[column]) and pd.api.types.is_numeric_dtype(dataframe_2[column]):
            stat, p_value = levene(dataframe_1[column].dropna(), dataframe_2[column].dropna())
            results[column] = {"levene_statistic": stat, "p_value": p_value}
    return results

def KS_test(dataframe_1, dataframe_2, column_name):
    common_columns = set(dataframe_1.columns).intersection(set(dataframe_2.columns))
    if not common_columns:
        raise ValueError("No common columns found between the two DataFrames.")
    results = {}
    for column in common_columns:
        if pd.api.types.is_numeric_dtype(dataframe_1[column]) and pd.api.types.is_numeric_dtype(dataframe_2[column]):
            stat, p_value = ks_2samp(dataframe_1[column].dropna(), dataframe_2[column].dropna())
            results[column] = {"ks_statistic": stat, "p_value": p_value}
    return results

def significant_diff(dataframe_1, dataframe_2, column_name, alpha=0.05):
    if KS_test(dataframe_1, dataframe_2, column_name)[column_name]["p_value"] < alpha:
        return True
    elif mean_test(dataframe_1, dataframe_2, column_name)[column_name]["p_value"] < alpha:
        return True
    elif variance_test(dataframe_1, dataframe_2, column_name)[column_name]["p_value"] < alpha:
        return True
    else:
        return False


# =============================================================================
# CAUSAL DISCOVERY EXTENSIONS
# =============================================================================

# -----------------------------------------------------------------------------
# Phase 1: Pairwise MI matrix
# -----------------------------------------------------------------------------

def compute_pairwise_MI(dataframe, method="kde"):
    """
    Compute a symmetric MI matrix for all numeric columns.

    Parameters
    ----------
    dataframe : pd.DataFrame
    method    : str — one of "kde", "binned", "histogram"

    Returns
    -------
    mi_matrix : pd.DataFrame  (symmetric, zeros on diagonal)
    """
    if method not in MI_METHODS:
        raise ValueError(f"Unknown method '{method}'. Choose from: {list(MI_METHODS)}")

    mi_fn = MI_METHODS[method]

    # Drop a pure index column if present (same logic as plot_mutual_information)
    df   = dataframe.iloc[:, 1:]
    cols = [c for c in get_columns(df) if pd.api.types.is_numeric_dtype(df[c])]
    n    = len(cols)

    mi_matrix = pd.DataFrame(np.zeros((n, n)), index=cols, columns=cols)

    for i in range(n):
        for j in range(i + 1, n):
            x = pd.to_numeric(df[cols[i]], errors="coerce").dropna().to_numpy()
            y = pd.to_numeric(df[cols[j]], errors="coerce").dropna().to_numpy()

            if len(x) < 10 or len(y) < 10 or np.std(x) == 0 or np.std(y) == 0:
                mi = 0.0
            else:
                mi = mi_fn(x, y)

            mi_matrix.loc[cols[i], cols[j]] = mi
            mi_matrix.loc[cols[j], cols[i]] = mi

    return mi_matrix


def plot_MI_heatmap(mi_matrix):
    """
    Plot the MI matrix as a colour-coded heatmap so high-MI pairs
    are immediately visible.
    """
    fig, ax = plt.subplots(figsize=(max(5, len(mi_matrix)), max(4, len(mi_matrix) - 1)))
    sns.heatmap(
        mi_matrix,
        annot=True, fmt=".3f",
        cmap="YlOrRd",
        linewidths=0.5,
        ax=ax,
        vmin=0,
    )
    ax.set_title("Pairwise Mutual Information Matrix")
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------
# Phase 2: Build initial skeleton via marginal MI threshold
# -----------------------------------------------------------------------------

def get_skeleton(mi_matrix, threshold=0.05):
    """
    Keep an edge (X, Y) only when MI(X, Y) > threshold.

    Parameters
    ----------
    mi_matrix : pd.DataFrame — output of compute_pairwise_MI
    threshold : float        — MI below this → no direct edge

    Returns
    -------
    adjacency : dict  {node: set of neighbours}
    removed   : list  of (node1, node2, reason) tuples
    """
    cols      = mi_matrix.columns.tolist()
    adjacency = {c: set() for c in cols}
    removed   = []

    for i, c1 in enumerate(cols):
        for j, c2 in enumerate(cols):
            if i >= j:
                continue
            if mi_matrix.loc[c1, c2] > threshold:
                adjacency[c1].add(c2)
                adjacency[c2].add(c1)
            else:
                removed.append((c1, c2, f"marginal MI={mi_matrix.loc[c1,c2]:.4f} ≤ {threshold}"))

    return adjacency, removed


# -----------------------------------------------------------------------------
# Phase 3: Conditional independence test & skeleton pruning
# -----------------------------------------------------------------------------

def conditional_MI(dataframe, col_x, col_y, col_z, n_splits=4, method="kde"):
    """
    Estimate MI(X ; Y | Z) by stratifying on Z and averaging MI per stratum.

    A value close to 0 means X ⊥ Y | Z (conditional independence).

    Parameters
    ----------
    dataframe : pd.DataFrame
    col_x     : str
    col_y     : str
    col_z     : str  — the conditioning variable
    n_splits  : int  — number of quantile-based strata for Z
    method    : str  — MI estimator

    Returns
    -------
    float : average MI across strata (lower → more independent)
    """
    mi_fn     = MI_METHODS.get(method, kde_MI)
    df        = dataframe[[col_x, col_y, col_z]].dropna()
    quantiles = np.linspace(0, 1, n_splits + 1)
    thresholds = df[col_z].quantile(quantiles).values

    mi_values = []
    for k in range(n_splits):
        lo, hi  = thresholds[k], thresholds[k + 1]
        mask    = (df[col_z] >= lo) & (df[col_z] <= hi)
        group   = df[mask]

        if len(group) < 15:
            continue

        x = group[col_x].to_numpy()
        y = group[col_y].to_numpy()

        if np.std(x) == 0 or np.std(y) == 0:
            mi_values.append(0.0)
        else:
            mi_values.append(mi_fn(x, y))

    return float(np.mean(mi_values)) if mi_values else 0.0


def conditional_independence_test(dataframe, col_x, col_y, col_z,
                                   n_splits=4, alpha=0.05):
    """
    Test X ⊥ Y | Z using both:
      - conditional MI (near-zero → independent)
      - significant_diff on stratified sub-groups

    Returns True if X and Y appear conditionally independent given Z.
    """
    df        = dataframe[[col_x, col_y, col_z]].dropna()
    quantiles = np.linspace(0, 1, n_splits + 1)
    thresholds = df[col_z].quantile(quantiles).values

    indep_flags = []

    for k in range(n_splits):
        lo, hi  = thresholds[k], thresholds[k + 1]
        mask    = (df[col_z] >= lo) & (df[col_z] <= hi)
        group   = df[mask].copy()

        if len(group) < 20:
            continue

        median_x  = group[col_x].median()
        low_group  = group[group[col_x] <  median_x]
        high_group = group[group[col_x] >= median_x]

        if len(low_group) < 5 or len(high_group) < 5:
            continue

        # Within this Z-stratum, does splitting on X change Y?
        dependent = significant_diff(low_group, high_group, col_y, alpha=alpha)
        indep_flags.append(not dependent)

    # Call independent only if all strata agree on independence
    return all(indep_flags) if indep_flags else False


def prune_skeleton_conditional(dataframe, adjacency, removed,
                                alpha=0.05, method="kde",
                                cmi_threshold=0.02):
    """
    PC-algorithm-style edge pruning:
    For each edge (X, Y), test X ⊥ Y | Z for every Z in neighbours(X) ∪ neighbours(Y).
    Remove the edge if any single Z makes them independent.

    Also records the *separation set* — the Z variable that caused removal —
    which is needed for v-structure detection.

    Parameters
    ----------
    dataframe     : pd.DataFrame
    adjacency     : dict  — from get_skeleton
    removed       : list  — from get_skeleton (will be extended in-place)
    alpha         : float — significance level for conditional_independence_test
    method        : str   — MI estimator for conditional_MI
    cmi_threshold : float — if cond-MI drops below this, also flag as independent

    Returns
    -------
    adjacency : dict  (pruned)
    removed   : list  (extended)
    sep_sets  : dict  {(X,Y): Z_that_separates}
    """
    cols     = list(adjacency.keys())
    sep_sets = {}

    for c1 in cols:
        for c2 in list(adjacency[c1]):
            if c2 <= c1:
                continue  # process each pair once

            # Candidate conditioning variables: all neighbours of c1 or c2
            candidates = (adjacency[c1] | adjacency[c2]) - {c1, c2}

            for cz in candidates:
                if cz not in dataframe.columns:
                    continue

                # Fast CMI check
                cmi = conditional_MI(dataframe, c1, c2, cz, method=method)

                ci_by_test = conditional_independence_test(
                    dataframe, c1, c2, cz, alpha=alpha
                )

                if cmi < cmi_threshold or ci_by_test:
                    adjacency[c1].discard(c2)
                    adjacency[c2].discard(c1)
                    sep_sets[(c1, c2)] = cz
                    sep_sets[(c2, c1)] = cz
                    removed.append(
                        (c1, c2,
                         f"cond. indep. given {cz}  (CMI={cmi:.4f}, test={ci_by_test})")
                    )
                    break  # edge removed — stop checking other Z's

    return adjacency, removed, sep_sets


# -----------------------------------------------------------------------------
# Phase 4: V-structure (collider) detection
# -----------------------------------------------------------------------------

def find_v_structures(dataframe, adjacency, sep_sets, alpha=0.05):
    """
    Identify colliders X → Z ← Y (v-structures / unshielded colliders).

    An unshielded triple is X - Z - Y where X and Y are NOT adjacent.
    Z is a collider if Z was NOT in the separation set of X and Y,
    because conditioning on a collider induces dependence (Berkson's paradox).

    Returns
    -------
    colliders    : list of (X, Z, Y) triples
    orientations : list of directed edges (X→Z) and (Y→Z)
    """
    cols         = list(adjacency.keys())
    colliders    = []
    orientations = []

    for cz in cols:
        neighbours = list(adjacency[cz])
        for i in range(len(neighbours)):
            for j in range(i + 1, len(neighbours)):
                cx, cy = neighbours[i], neighbours[j]

                # Must be an unshielded triple: X and Y not adjacent
                if cy in adjacency[cx]:
                    continue

                # Z is a collider iff Z ∉ sep(X, Y)
                sep = sep_sets.get((cx, cy), sep_sets.get((cy, cx), None))
                if sep != cz:
                    colliders.append((cx, cz, cy))
                    orientations.append((cx, cz))
                    orientations.append((cy, cz))

    # Deduplicate orientations
    orientations = list(set(orientations))
    return colliders, orientations


# -----------------------------------------------------------------------------
# Phase 5: Meek orientation rules
# -----------------------------------------------------------------------------

def apply_meek_rules(adjacency, orientations):
    """
    Propagate known edge orientations using Meek's rules to orient
    as many undirected edges as possible without new independence tests.

    Rules applied:
      R1: A → B — C,  A not adj C  →  B → C   (avoid new collider)
      R2: A → C,  A — B — C        →  A → B   (avoid cycle)
      R3: A — B,  A → C ← B        →  A → B

    Returns
    -------
    directed   : set of (from, to) tuples
    undirected : set of frozenset({X, Y}) — edges still without orientation
    """
    directed   = set(orientations)
    undirected = set()

    for node, neighbours in adjacency.items():
        for nb in neighbours:
            edge = (node, nb)
            rev  = (nb, node)
            if edge not in directed and rev not in directed:
                undirected.add(frozenset({node, nb}))

    changed = True
    while changed:
        changed = False

        for fs in list(undirected):
            u, v = tuple(fs)

            # R1 — check both orientations
            for (src, tgt) in [(u, v), (v, u)]:
                # If some Z → src exists, and tgt not adj Z → orient src → tgt
                for z in adjacency[src]:
                    if (z, src) in directed and tgt not in adjacency[z]:
                        directed.add((src, tgt))
                        undirected.discard(fs)
                        changed = True
                        break
                if fs not in undirected:
                    break

            if fs not in undirected:
                continue

            # R2 — If src → tgt exists via a directed path through a common nb
            for (src, tgt) in [(u, v), (v, u)]:
                for nb in adjacency[src]:
                    if (src, nb) in directed and (nb, tgt) in directed:
                        directed.add((src, tgt))
                        undirected.discard(fs)
                        changed = True
                        break
                if fs not in undirected:
                    break

    return directed, undirected


# -----------------------------------------------------------------------------
# Phase 6: Hidden confounder (_H) candidate detection
# -----------------------------------------------------------------------------

def find_confounder_candidates(mi_matrix, adjacency, threshold=0.05):
    """
    Identify pairs that are correlated (high MI) but have NO direct edge
    in the pruned skeleton — a hallmark of hidden confounding by _H.

    Parameters
    ----------
    mi_matrix : pd.DataFrame
    adjacency : dict — pruned skeleton
    threshold : float — minimum MI to flag a pair

    Returns
    -------
    candidates : list of (X, Y, MI_value) sorted by MI descending
    """
    cols       = mi_matrix.columns.tolist()
    candidates = []

    for i, c1 in enumerate(cols):
        for j, c2 in enumerate(cols):
            if i >= j:
                continue
            if c2 not in adjacency[c1]:          # no direct edge
                mi = mi_matrix.loc[c1, c2]
                if mi > threshold:               # but still correlated
                    candidates.append((c1, c2, mi))

    return sorted(candidates, key=lambda x: -x[2])


# -----------------------------------------------------------------------------
# Phase 7: Intervention effect analysis (for later email batches)
# -----------------------------------------------------------------------------

def intervention_effect(obs_df, int_df, target_col, alpha=0.05):
    """
    Test whether an intervention changed the distribution of target_col.

    Parameters
    ----------
    obs_df     : pd.DataFrame — observational samples
    int_df     : pd.DataFrame — samples collected under intervention on some node
    target_col : str          — column to check for distributional shift
    alpha      : float

    Returns
    -------
    bool : True if intervention significantly changed target_col
    """
    return significant_diff(obs_df, int_df, target_col, alpha=alpha)


def intervention_effect_all_columns(obs_df, int_df, alpha=0.05):
    """
    Run intervention_effect for every numeric column shared between
    obs_df and int_df.  Useful for quickly seeing which nodes responded
    to a particular intervention.

    Returns
    -------
    pd.DataFrame with columns: variable, significant, ks_p, t_p, levene_p
    """
    common = [
        c for c in obs_df.columns
        if c in int_df.columns and pd.api.types.is_numeric_dtype(obs_df[c])
    ]
    rows = []
    for col in common:
        ks_p  = KS_test(obs_df, int_df, col)[col]["p_value"]
        t_p   = mean_test(obs_df, int_df, col)[col]["p_value"]
        lev_p = variance_test(obs_df, int_df, col)[col]["p_value"]
        sig   = (ks_p < alpha) or (t_p < alpha) or (lev_p < alpha)
        rows.append({
            "variable":    col,
            "significant": sig,
            "ks_p":        round(ks_p,  4),
            "t_p":         round(t_p,   4),
            "levene_p":    round(lev_p, 4),
        })

    if not rows:
        return pd.DataFrame(
            columns=["variable", "significant", "ks_p", "t_p", "levene_p"]
        )

    return pd.DataFrame(rows).sort_values("ks_p")


def update_dag_with_interventions(obs_df, results,
                                 intervention_dirs=("./data", "./testdata"),
                                 alpha=0.05, verbose=True):
    """
    Use intervention datasets to orient additional edges in an inferred DAG.

    For a file like '-2a.csv' or '2d.csv', the trailing letters are treated as
    the intervened variable. Any significant downstream shifts are oriented as
    target -> affected_variable.
    """
    if not intervention_dirs:
        results.setdefault("interventions", [])
        return results

    directed = set(results.get("directed", set()))
    undirected = set(results.get("undirected", set()))
    adjacency = {
        node: set(neighbours)
        for node, neighbours in results.get("adjacency", {}).items()
    }
    intervention_results = []

    for file_path in discover_intervention_files(
        intervention_dirs=intervention_dirs,
        exclude_files=["data_2358.csv"]
    ):
        target_col = parse_intervention_target(file_path.name, obs_df.columns)
        if target_col is None or target_col not in obs_df.columns:
            continue

        int_df = pd.read_csv(file_path)
        effects = intervention_effect_all_columns(obs_df, int_df, alpha=alpha)
        if effects.empty:
            continue

        if verbose:
            print(f"\nAnalyserer intervention: {file_path.name} (target={target_col})")

        significant_effects = effects[
            (effects["significant"]) & (effects["variable"] != target_col)
        ]

        for _, row in significant_effects.iterrows():
            effect_col = row["variable"]
            adjacency.setdefault(target_col, set()).add(effect_col)
            adjacency.setdefault(effect_col, set()).add(target_col)

            edge = frozenset({target_col, effect_col})
            undirected.discard(edge)
            directed.discard((effect_col, target_col))
            directed.add((target_col, effect_col))

            intervention_results.append({
                "file": file_path.name,
                "target": target_col,
                "effect": effect_col,
                "ks_p": row["ks_p"],
                "t_p": row["t_p"],
                "levene_p": row["levene_p"],
            })

            if verbose:
                print(f"  Orienteret fra intervention: {target_col} -> {effect_col}")

    directed, undirected = apply_meek_rules(adjacency, directed)

    results["adjacency"] = adjacency
    results["directed"] = directed
    results["undirected"] = undirected
    results["interventions"] = intervention_results
    return results


# -----------------------------------------------------------------------------
# Visualisation: skeleton & DAG
# -----------------------------------------------------------------------------

def plot_skeleton(adjacency, directed=None, undirected=None, title="Graph Skeleton"):
    """
    Simple text-based adjacency summary + optional matplotlib network plot.

    directed   : set of (from, to) tuples  — drawn as arrows
    undirected : set of frozensets          — drawn as plain lines
    """
    try:
        import networkx as nx
        G = nx.DiGraph()
        nodes = list(adjacency.keys())
        G.add_nodes_from(nodes)

        if directed:
            for (u, v) in directed:
                G.add_edge(u, v, style="directed")
        if undirected:
            for fs in undirected:
                u, v = tuple(fs)
                G.add_edge(u, v, style="undirected")
                G.add_edge(v, u, style="undirected")

        pos = nx.spring_layout(G, seed=42)
        fig, ax = plt.subplots(figsize=(7, 5))

        directed_edges   = [(u, v) for u, v, d in G.edges(data=True)
                            if d.get("style") == "directed"]
        undirected_edges = [(u, v) for u, v, d in G.edges(data=True)
                            if d.get("style") == "undirected"]

        nx.draw_networkx_nodes(G, pos, node_size=800, node_color="steelblue",
                               alpha=0.9, ax=ax)
        nx.draw_networkx_labels(G, pos, font_color="white", font_size=10, ax=ax)
        nx.draw_networkx_edges(G, pos, edgelist=directed_edges,
                               arrows=True, arrowsize=20,
                               edge_color="navy", width=2, ax=ax)
        nx.draw_networkx_edges(G, pos, edgelist=undirected_edges,
                               arrows=False,
                               edge_color="grey", width=1.5,
                               style="dashed", ax=ax)
        ax.set_title(title)
        ax.axis("off")
        plt.tight_layout()
        plt.show()

    except ImportError:
        # Fallback: plain text summary
        print(f"\n=== {title} ===")
        for node, neighbours in adjacency.items():
            print(f"  {node}: {sorted(neighbours)}")
        if directed:
            print("\nDirected edges:")
            for e in sorted(directed):
                print(f"  {e[0]} → {e[1]}")
        if undirected:
            print("\nUndirected edges (need interventions):")
            for fs in undirected:
                u, v = tuple(fs)
                print(f"  {u} — {v}")


# -----------------------------------------------------------------------------
# Master runner
# -----------------------------------------------------------------------------

def infer_dag(dataframe, mi_threshold=0.05, alpha=0.05,
              method="kde", cmi_threshold=0.02,
              intervention_dirs=None, verbose=True):
    """
    Full observational causal-discovery pipeline.

    Steps
    -----
    1. Compute pairwise MI  →  identify correlated pairs
    2. Build initial skeleton via MI threshold
    3. Prune skeleton with conditional independence tests (PC-style)
    4. Detect v-structures (colliders) from separation sets
    5. Apply Meek rules to orient remaining edges
    6. Flag hidden-confounder (_H) candidates

    Parameters
    ----------
    dataframe     : pd.DataFrame — observational data
    mi_threshold  : float — MI below this → no direct edge in skeleton
    alpha         : float — significance level for CI tests
    method        : str   — MI estimator: "kde" | "binned" | "histogram"
    cmi_threshold : float — CMI below this → conditionally independent
    verbose       : bool  — print progress

    Returns
    -------
    results : dict with keys
        mi_matrix, adjacency, directed, undirected,
        colliders, confounder_candidates, removed, sep_sets
        and interventions when intervention_dirs is provided
    """

    def log(msg):
        if verbose:
            print(msg)

    log("=" * 60)
    log("STEP 1 — Pairwise Mutual Information")
    log("=" * 60)
    mi_matrix = compute_pairwise_MI(dataframe, method=method)
    log(mi_matrix.round(3).to_string())
    plot_MI_heatmap(mi_matrix)

    log("\n" + "=" * 60)
    log("STEP 2 — Initial Skeleton  (threshold={})".format(mi_threshold))
    log("=" * 60)
    adjacency, removed = get_skeleton(mi_matrix, threshold=mi_threshold)
    log("Edges kept:")
    for node, nbs in adjacency.items():
        if nbs:
            log(f"  {node}: {sorted(nbs)}")
    log(f"Edges removed: {len(removed)}")

    log("\n" + "=" * 60)
    log("STEP 3 — Conditional Independence Pruning")
    log("=" * 60)
    adjacency, removed, sep_sets = prune_skeleton_conditional(
        dataframe, adjacency, removed,
        alpha=alpha, method=method, cmi_threshold=cmi_threshold
    )
    log("Pruned skeleton:")
    for node, nbs in adjacency.items():
        log(f"  {node}: {sorted(nbs)}")
    log(f"\nSeparation sets found: {len(sep_sets) // 2}")
    for (a, b), z in sep_sets.items():
        if a < b:
            log(f"  sep({a},{b}) = {z}")

    log("\n" + "=" * 60)
    log("STEP 4 — V-Structure Detection")
    log("=" * 60)
    colliders, orientations = find_v_structures(
        dataframe, adjacency, sep_sets, alpha=alpha
    )
    if colliders:
        for (cx, cz, cy) in colliders:
            log(f"  Collider: {cx} → {cz} ← {cy}")
    else:
        log("  No colliders detected from observational data alone.")

    log("\n" + "=" * 60)
    log("STEP 5 — Meek Orientation Rules")
    log("=" * 60)
    directed, undirected = apply_meek_rules(adjacency, orientations)
    log("Confidently directed edges:")
    for (u, v) in sorted(directed):
        log(f"  {u} → {v}")
    log("Still undirected (require interventions to orient):")
    for fs in undirected:
        u, v = tuple(fs)
        log(f"  {u} — {v}")

    if intervention_dirs:
        log("\n" + "=" * 60)
        log("STEP 5B — Intervention-based Orientation")
        log("=" * 60)
        results = {
            "mi_matrix":              mi_matrix,
            "adjacency":              adjacency,
            "directed":               directed,
            "undirected":             undirected,
            "colliders":              colliders,
            "confounder_candidates":  [],
            "removed":                removed,
            "sep_sets":               sep_sets,
        }
        results = update_dag_with_interventions(
            dataframe,
            results,
            intervention_dirs=intervention_dirs,
            alpha=alpha,
            verbose=verbose,
        )
        adjacency = results["adjacency"]
        directed = results["directed"]
        undirected = results["undirected"]
        log("Intervention-based directed edges:")
        for (u, v) in sorted(directed):
            log(f"  {u} → {v}")
        log(f"Intervention edges added: {len(results.get('interventions', []))}")

    log("\n" + "=" * 60)
    log("STEP 6 — Hidden Confounder (_H) Candidates")
    log("=" * 60)
    hc = find_confounder_candidates(mi_matrix, adjacency, threshold=mi_threshold)
    if hc:
        log("Pairs correlated but without a direct edge (likely confounded by _H):")
        for (c1, c2, mi) in hc:
            log(f"  {c1} <-> {c2}  (MI={mi:.4f})")
    else:
        log("  No strong hidden-confounder candidates found.")

    log("\n" + "=" * 60)
    log("SUMMARY")
    log("=" * 60)
    log(f"  Directed edges   : {len(directed)}")
    log(f"  Undirected edges : {len(undirected)}  ← target for interventions")
    log(f"  Colliders found  : {len(colliders)}")
    log(f"  _H candidates    : {len(hc)}")
    if intervention_dirs:
        log(f"  Intervention edges: {len(results.get('interventions', []))}")
    log(f"  Total edges removed from full graph: {len(removed)}")

    plot_skeleton(adjacency, directed=directed, undirected=undirected,
                  title="Inferred Skeleton (blue=directed, grey dashed=undirected)")

    return {
        "mi_matrix":              mi_matrix,
        "adjacency":              adjacency,
        "directed":               directed,
        "undirected":             undirected,
        "colliders":              colliders,
        "confounder_candidates":  hc,
        "removed":                removed,
        "sep_sets":               sep_sets,
        "interventions":          results.get("interventions", []) if intervention_dirs else [],
    }


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    df = load_csv_data("data_2358.csv")

    if df is not None:
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=['Unnamed: 0'])
        # --- Step 0: quick look ---
        print(df.head())
        print(df.describe())

        # --- Step 1: visual exploration ---
        #plot_mutual_information(df, method="kde")

        # --- Step 2: full causal discovery pipeline ---
        results = infer_dag(
            df,
            mi_threshold  = 0.045,
            alpha         = 0.05,
            method        = "kde",
            cmi_threshold = 0.02,
            intervention_dirs = ("./data",),
            verbose       = True,
        )

        # --- Step 3: once you have an interventional dataset ---
        # int_df = load_csv_data("intervention_A1.csv")
        # if int_df is not None:
        #     print(intervention_effect_all_columns(df, int_df))