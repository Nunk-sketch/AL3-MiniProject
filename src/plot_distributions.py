import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

data_dir = './data'
csv_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.csv')])

intervention_groups = {}
baselines = {}

for f in csv_files:
    if f == '0ec.csv':
        continue

    df = pd.read_csv(os.path.join(data_dir, f)).iloc[:, 1:].select_dtypes(include=['number'])
    
    match = re.search(r'([a-zA-Z])\.csv$', f)
    if match:
        letter = match.group(1).upper()
        if letter not in intervention_groups:
            intervention_groups[letter] = []
        intervention_groups[letter].append(df)
    else:
        baselines[f] = df

combined_groups = {k: pd.concat(v) for k, v in intervention_groups.items()}

cols_to_plot = ['A', 'B', 'C', 'D', 'E', 'F']
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
axes = axes.flatten()

for i, col in enumerate(cols_to_plot):
    ax = axes[i]
    
    for letter, data in combined_groups.items():
        if letter != col:
            sns.kdeplot(data[col], label=f"Intervention on {letter}", ax=ax, alpha=0.6)
            
    for fname, df in baselines.items():
        if col in df.columns:
            sns.kdeplot(df[col], label=fname, ax=ax, color='black', linewidth=2.5)

    ax.set_title(f'Column {col} (Excluding {col} Interventions)', fontsize=14, fontweight='bold')
    ax.legend(fontsize='x-small', ncol=2)
    ax.set_ylabel('Density')

plt.tight_layout()
plt.show()