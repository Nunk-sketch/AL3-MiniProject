import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def pairwise_linear_correlations(df: pd.DataFrame) -> pd.DataFrame:
	numeric_df = df.select_dtypes(include=["number"])
	cols = list(numeric_df.columns)

	rows = []
	for i in range(len(cols)):
		for j in range(i + 1, len(cols)):
			col_a = cols[i]
			col_b = cols[j]
			corr = numeric_df[col_a].corr(numeric_df[col_b], method="pearson")
			rows.append({"var1": col_a, "var2": col_b, "pearson_r": corr})

	result = pd.DataFrame(rows).sort_values(by="pearson_r", key=lambda s: s.abs(), ascending=False)
	return result


df = pd.read_csv(r"data\data_1504.csv")
correlations = pairwise_linear_correlations(df)
print(correlations.to_string(index=False))
dflog = df.copy()
for col in dflog.columns:
    if dflog[col].dtype in ["int64", "float64"]:
        dflog[col] = np.log(dflog[col] + 1e-9)
correlations_log = pairwise_linear_correlations(dflog)
print(correlations_log.to_string(index=False))

sns.pairplot(dflog)
plt.show()