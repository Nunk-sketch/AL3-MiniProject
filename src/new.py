import pandas as pd
import glob
import os
import re


THRESHOLD = 0.25
CORR_THRESHOLD = 0.4


def load_obs(path):
    df = pd.read_csv(path)

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    return df


def load_interventions(folder):
    files = glob.glob(os.path.join(folder, "*.csv"))
    interventions = {}

    for f in files:
        name = os.path.basename(f).replace(".csv", "")

        if "data_2358" in name:
            continue 

        df = pd.read_csv(f)

        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])

        interventions[name] = df
    return interventions


def get_variable_from_name(name):
    match = re.search(r'[a-zA-Z]', name[::-1])
    return match.group().upper() if match else name


def compute_effects(obs_means, interventions):
    effects = {}
    for name, df in interventions.items():
        delta = (df.mean() - obs_means).abs() 
        source = get_variable_from_name(name)

        if source not in effects:
            effects[source] = []
        effects[source].append(delta)
    
    avg_effects = {k: pd.concat(v, axis=1).mean(axis=1) for k, v in effects.items()}
    return avg_effects


def detect_edges(effects, threshold=THRESHOLD):
    edges = set()
    sources = list(effects.keys())
    
    for source in sources:
        for target, val in effects[source].items():
            if source == target:
                continue
            
            forward_effect = val
            backward_effect = effects.get(target, pd.Series({source: 0})).get(source, 0)
            
            if forward_effect > threshold and forward_effect > backward_effect:
                edges.add((source, target))
    return edges


def find_confounder_pair(corr, edges, threshold=CORR_THRESHOLD):
    pairs = []
    nodes = list(corr.columns)

    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            u, v = nodes[i], nodes[j]
            
            if abs(corr.loc[u, v]) > threshold:
                if (u, v) not in edges and (v, u) not in edges:
                    pairs.append((u, v))
    return pairs


if __name__ == "__main__":
    folder = "./data"
    obs_df = load_obs(os.path.join(folder, "data_2358.csv"))
    interventions = load_interventions(folder)
    
    obs_means = obs_df.mean()
    corr = obs_df.corr()
    
    avg_effects = compute_effects(obs_means, interventions)
    
    matrix = pd.DataFrame(avg_effects).transpose()
    matrix = matrix.sort_index().reindex(sorted(matrix.columns), axis=1)
    
    print("Average Absolute Effect Matrix:")
    print(matrix.round(3))
    print("----------------------------------------")
    
    edges = detect_edges(avg_effects)
    
    all_vars = set(obs_df.columns)
    intervened_vars = set(avg_effects.keys())
    C_candidates = all_vars - intervened_vars
    
    conf_pairs = find_confounder_pair(corr, edges)

    print("Causal Edges:")
    for e in sorted(edges):
        print(f"{e[0]} -> {e[1]}")

    print("\nCandidate C Node:")
    print(f"C = {list(C_candidates)}")

    print("\nConfounded Pairs (_H):")
    for p in conf_pairs:
        print(f"{p[0]} <- _H -> {p[1]}")