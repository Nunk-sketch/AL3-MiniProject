import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Visualisering af korrelation (Observationel data)
def plot_correlations(df):
    corr = df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title("Korrelationsmatrix (Observationel)")
    plt.show()

# Sammenlign Intervention vs Observation
# Hvis fordelingen af B er ens i begge, men de er korrelerede i obs_df, så er der en confounder
def check_confounder(obs_df, int_df_a, var_target='B'):
    plt.figure(figsize=(10, 5))
    sns.kdeplot(obs_df[var_target], label='Observationel', shade=True)
    sns.kdeplot(int_df_a[var_target], label=f'Intervention på A', shade=True)
    plt.legend()
    plt.title(f"Fordeling af {var_target}: Obs vs Int")
    plt.show()

df = pd.read_csv(r"data\data_1504.csv")

plot_correlations(df)
check_confounder(df, df[df['Intervention'] == 'A'], var_target='B')