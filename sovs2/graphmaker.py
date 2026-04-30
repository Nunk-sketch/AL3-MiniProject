import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import mutual_info_regression
import os
import glob

# ==========================================
# 1. KONFIGURATION OG LOADER
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'data'))
BASELINE_FILE = 'data_2358.csv'

def load_datasets():
    files = glob.glob(os.path.join(DATA_DIR, '*.csv'))
    loaded = {}
    for p in sorted(files):
        name = os.path.basename(p)
        try:
            df = pd.read_csv(p)
        except Exception as e:
            print(f"Kunne ikke læse {name}: {e}")
            continue
        if df.shape[1] > 1:
            df = df.iloc[:, 1:] # Drop index kolonne
        loaded[name] = df

    # Vælg baseline: foretrukket fil eller fallback til første fundne
    baseline_df = loaded.get(BASELINE_FILE)
    if baseline_df is None and len(loaded) > 0:
        first_key = sorted(loaded.keys())[0]
        baseline_df = loaded[first_key]
        print(f"Baseline '{BASELINE_FILE}' ikke fundet — bruger fallback: {first_key}")

    # Organiser i specifikke grupper
    data = {
        'baseline': baseline_df,
        'int_A_pos': loaded.get('2a.csv'),
        'int_A_neg': loaded.get('-2a.csv'),
        'int_E_pos': loaded.get('2e.csv'),
        'int_A_neg_fallback': loaded.get('-2a.csv')
    }
    return data

# ==========================================
# 2. KAUSALE ANALYSE-FUNKTIONER
# ==========================================

def get_r2_diff(var1, var2, df):
    """Beregner asymmetri i R2: R2(var1->var2) - R2(var2->var1)"""
    d = df[[var1, var2]].dropna()
    if len(d) < 10:
        return 0

    # Model 1: var1 -> var2
    r2_12 = LinearRegression().fit(d[[var1]], d[var2]).score(d[[var1]], d[var2])
    # Model 2: var2 -> var1
    r2_21 = LinearRegression().fit(d[[var2]], d[var1]).score(d[[var2]], d[var1])

    return r2_12 - r2_21

def calculate_ace(df_high, df_low, target):
    """Estimerer Average Causal Effect (ACE)"""
    if df_high is None or df_low is None:
        return 0
    if target not in df_high.columns or target not in df_low.columns:
        return 0
    return (df_high[target].mean() - df_low[target].mean()) / 4

def run_discovery():
    datasets = load_datasets()
    df_base = datasets.get('baseline')
    if df_base is None:
        raise FileNotFoundError(f"Baseline fil '{BASELINE_FILE}' ikke fundet i {DATA_DIR}")
    cols = df_base.select_dtypes(include=np.number).columns.tolist()
    if len(cols) < 2:
        raise ValueError("Mindst to numeriske kolonner kræves i baseline for at køre discovery.")
    
    G = nx.DiGraph()
    G.add_nodes_from(cols)
    
    print(f"{'Par':<15} | {'MI':<6} | {'ACE(A)':<8} | {'R2 Diff':<8} | {'Konklusion'}")
    print("-" * 65)

    confounder_candidates = []
    
    for i, var1 in enumerate(cols):
        for var2 in cols[i+1:]:
            # 1. Mutual Information (Styrke af forbindelse)
            # Mutual information kan fejle hvis for få samples eller konstante kolonner
            try:
                mi_val = mutual_info_regression(df_base[[var1]], df_base[var2], random_state=42)
                mi = float(mi_val[0]) if len(mi_val) > 0 else 0.0
            except Exception:
                mi = 0.0

            # Ignorer svage forbindelser
            if mi < 0.05:
                continue
            
            # 2. ACE fra A (Flytter A denne variabel?)
            # Vi tjekker om intervention på A påvirker var1 og var2
            ace_a_v1 = calculate_ace(datasets.get('int_A_pos'), datasets.get('int_A_neg'), var1)
            ace_a_v2 = calculate_ace(datasets.get('int_A_pos'), datasets.get('int_A_neg'), var2)
            
            # 3. R2 Asymmetri (Retning)
            r2_diff = get_r2_diff(var1, var2, df_base)
            
            conclusion = "Uafklaret"
            
            # LOGIK FOR CONFOUNDER H: 
            # Hvis de har høj MI, men ACE fra A er 0 for begge (eller de er uafhængige af A)
            # og R2 diff er tæt på 0.
            if abs(r2_diff) < 0.05 and mi > 0.4:
                # Tjek om de er påvirket af A. Hvis ACE er meget lille, er de styret af H.
                if abs(ace_a_v1) < 0.01 and abs(ace_a_v2) < 0.01:
                    conclusion = "Skjult Confounder (H)"
                    confounder_candidates.append((var1, var2, mi))
            
            # LOGIK FOR DIREKTE KANT:
            elif abs(r2_diff) > 0.08:
                if r2_diff > 0:
                    conclusion = f"{var1} -> {var2}"
                    G.add_edge(var1, var2)
                else:
                    conclusion = f"{var2} -> {var1}"
                    G.add_edge(var2, var1)

            print(f"{var1+'-'+var2:<15} | {mi:<6.2f} | {abs(ace_a_v1):<8.4f} | {r2_diff:<8.3f} | {conclusion}")

    # Tilføj den stærkeste confounder fundet
    if confounder_candidates:
        # Vælg den med højest MI
        best_h = max(confounder_candidates, key=lambda x: x[2])
        v1, v2, _ = best_h
        G.add_node('H')
        # Tilføj confounder-edges uden pile (mærket i data)
        G.add_edge('H', v1)
        G.add_edge('H', v2)
        print(f"\nPlaceret H mellem {v1} og {v2} baseret på høj MI og lav ACE.")

    # Hvis ingen edges blev fundet, lav en fallback ved at forbinde noder
    if G.number_of_edges() == 0:
        print("Ingen kausale kanter fundet — anvender fallback via baseline-korrelationer for at forbinde noder.")
        # Beregn absolutte Pearson-korrelationer på baseline
        corr = df_base.corr().abs().fillna(0)
        # Byg et fuldt vægtet undirected graph for MST
        U = nx.Graph()
        U.add_nodes_from(cols)
        for i, a in enumerate(cols):
            for b in cols[i+1:]:
                w = float(corr.loc[a, b]) if a in corr.columns and b in corr.columns else 0.0
                U.add_edge(a, b, weight=w)
        # Maksimal spanning tree
        try:
            T = nx.maximum_spanning_tree(U)
        except Exception:
            T = U
        # Tilføj kanter til G med retning bestemt af R2_asymmetri
        for u, v, d in T.edges(data=True):
            r2_diff = get_r2_diff(u, v, df_base)
            if r2_diff > 0:
                G.add_edge(u, v)
            else:
                G.add_edge(v, u)
        print("Fallback-forbindelser tilføjet (MST baseret på baseline-korrelation).")

    return G

# ==========================================
# 3. VISUALISERING
# ==========================================

def plot_graph(G):
    plt.figure(figsize=(12,8))

    # Brug undirected layout for sammenhængende placering, men behold retninger i G
    UG = G.to_undirected()
    try:
        pos = nx.kamada_kawai_layout(UG)
    except Exception:
        pos = nx.spring_layout(UG, k=0.6, iterations=200, seed=42)

    # Node farver
    node_colors = ['#FF6B6B' if n == 'H' else '#4ECDC4' for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_size=2000, node_color=node_colors)
    nx.draw_networkx_labels(G, pos, font_weight='bold')

    # Skil kanter ad
    confounder_edges = []
    standard_edges = []
    for u, v in G.edges():
        if u == 'H' or v == 'H':
            confounder_edges.append((u, v))
        else:
            standard_edges.append((u, v))

    # Tegn normale kanter med pile
    if standard_edges:
        nx.draw_networkx_edges(G, pos, edgelist=standard_edges, width=2, arrows=True, arrowstyle='-|>', arrowsize=20, connectionstyle='arc3,rad=0.08')

    # Tilføj kant-etiketter med en pil-tekst
    if standard_edges:
        edge_labels = {(u,v): '→' for u,v in standard_edges}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='black', font_size=12)

    # Tegn confounder kanter stiplede uden pile
    if confounder_edges:
        nx.draw_networkx_edges(G, pos, edgelist=confounder_edges, width=2, style='dashed', edge_color='red', arrows=False)

    # Gem som både SVG og PNG for god pile-visualisering
    out_svg = 'final_causal_model.svg'
    out_png = 'final_causal_model.png'
    plt.title('Causal Discovery: Identificerede forbindelser og skjult confounder (H)')
    plt.axis('off')
    plt.savefig(out_svg, format='svg', bbox_inches='tight')
    plt.savefig(out_png, dpi=200, bbox_inches='tight')
    print(f'Saved {out_svg} and {out_png}')
    plt.show()

if __name__ == "__main__":
    print("Starter Kausal Opdagelse...")
    final_graph = run_discovery()
    plot_graph(final_graph)
    print("\nAnalyse færdig. Graf gemt som 'final_causal_model.png'.")