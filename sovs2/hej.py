from pathlib import Path
import numpy as np
from scipy.stats import gaussian_kde
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable

def get_csv_file_list():
    data_dir = "./data"
    print(data_dir)
    return [str(path) for path in Path(data_dir).glob("*.csv")]

def get_csv_file(csv_file_name):
    tempList = get_csv_file_list()
    for file in tempList:
        if csv_file_name in file:
            return file
    return None

def load_csv_data(csv_file_name):
    csv_file_path = get_csv_file(csv_file_name)
    if csv_file_path:
        return pd.read_csv(csv_file_path)
    return None

def get_columns(dataframe):
    return dataframe.columns.tolist()


# ── MI Estimators ──────────────────────────────────────────────────────────────

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


# ── Method registry ────────────────────────────────────────────────────────────

MI_METHODS = {
    "kde":      kde_MI,
    "binned":   binned_MI,
    "histogram": histogram_MI,
}


# ── Plotting ───────────────────────────────────────────────────────────────────

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
                mi = mi_fn(x, y)   # ← uses whichever estimator was chosen

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

            ax_top.plot(x_range,         x_kde(x_range), color="steelblue", linewidth=1.5)
            ax_right.plot(y_kde(y_range), y_range,        color="steelblue", linewidth=1.5)
            ax_top.axis("off")
            ax_right.axis("off")

            ax.set_xlabel(cols[j], fontsize=8)
            ax.set_ylabel(cols[i], fontsize=8)
            ax.tick_params(labelsize=6)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    df = load_csv_data("data_2358.csv")

    # Swap method= to compare estimators:
    plot_mutual_information(df, method="kde")
    plot_mutual_information(df, method="binned")
    plot_mutual_information(df, method="histogram")