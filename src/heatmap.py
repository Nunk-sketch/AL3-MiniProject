import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_correlation_heatmap(csv_path: str = r"data\data_1504.csv") -> None:
    df = pd.read_csv(csv_path)

    # Keep only numeric columns and skip auto-generated index-like columns.
    numeric_df = df.select_dtypes(include=["number"])
    numeric_df = numeric_df.loc[:, ~numeric_df.columns.str.contains(r"^Unnamed")]

    corr = numeric_df.corr(method="pearson")

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "Pearson correlation"},
    )
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_correlation_heatmap()
