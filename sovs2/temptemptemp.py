import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

data_dir = './data'
csv_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.csv')])

# 1. Group files by their intervention target
intervention_groups = {}
baselines = {}

for f in csv_files:
    df = pd.read_csv(os.path.join(data_dir, f)).iloc[:, 1:].select_dtypes(include=['number'])
    
    # Extract the letter suffix (e.g., 'a' from '-2a.csv')
    match = re.search(r'([a-zA-Z])\.csv$', f)
    if match:
        letter = match.group(1).upper()
        if letter not in intervention_groups:
            intervention_groups[letter] = []
        intervention_groups[letter].append(df)
    else:
        baselines[f] = df

# Combine grouped dataframes
combined_groups = {k: pd.concat(v) for k, v in intervention_groups.items()}

# 2. Plotting with the exclusion filter
cols_to_plot = ['A', 'B', 'C', 'D', 'E', 'F']
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
axes = axes.flatten()

for i, col in enumerate(cols_to_plot):
    ax = axes[i]
    
    # EXCLUSION RULE: Only plot intervention groups that DID NOT target this column
    for letter, data in combined_groups.items():
        if letter != col:
            sns.kdeplot(data[col], label=f"Intervention on {letter}", ax=ax, alpha=0.6)
            
    # Always include baseline (e.g., data_2358.csv) in black for reference
    for fname, df in baselines.items():
        if col in df.columns:
            sns.kdeplot(df[col], label=fname, ax=ax, color='black', linewidth=2.5)

    ax.set_title(f'Column {col} (Excluding {col} Interventions)', fontsize=14, fontweight='bold')
    ax.legend(fontsize='x-small', ncol=2)
    ax.set_ylabel('Density')

plt.tight_layout()
plt.show()