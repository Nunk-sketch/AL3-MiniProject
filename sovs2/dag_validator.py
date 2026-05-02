# %% [markdown]
# # DAG Structure Validator
#
# Uses your observational + interventional datasets to check whether a proposed
# DAG is consistent with the data.
#
# **How to use:**
# 1. Run cells 0–2 (auto — loads data and plots exploration)
# 2. **Edit cell 3** to propose your DAG
# 3. Run cells 4–8 to see pass/fail evidence for every claim the graph makes
#
# Variables: A, B, C, D, E, F  |  Hidden confounder: _H

# %% [markdown]
# ## 0. Setup

# %%
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import networkx as nx
from itertools import permutations
import warnings

warnings.filterwarnings("ignore")
plt.rcParams["figure.figsize"] = (11, 5)
plt.rcParams["font.size"] = 11

ALPHA = 0.05   # significance threshold — change if you want stricter/looser tests
DATA_DIR = "./data"

# %% [markdown]
# ## 1. Load Data

# %%
obs = pd.read_csv(f"{DATA_DIR}/data_2358.csv", index_col=0)
VARS = list(obs.columns)   # ['A','B','C','D','E','F']
N = len(obs)

print(f"Observed variables : {VARS}")
print(f"Observational rows : {N}")
print("\nFirst 3 rows:")
display(obs.head(3))

# --- intervention files -------------------------------------------------
# Format: (intervened_variable, set_value) → DataFrame
int_data = {
    ("A", -2): pd.read_csv(f"{DATA_DIR}/-2a.csv", index_col=0),
    ("A",  2): pd.read_csv(f"{DATA_DIR}/2a.csv",  index_col=0),
    ("B", -2): pd.read_csv(f"{DATA_DIR}/-2b.csv", index_col=0),
    ("D",  2): pd.read_csv(f"{DATA_DIR}/2d.csv",  index_col=0),
    ("E",  2): pd.read_csv(f"{DATA_DIR}/2e.csv",  index_col=0),
    ("F", -2): pd.read_csv(f"{DATA_DIR}/-2f.csv", index_col=0),
}

print("\nIntervention datasets:")
for (v, val), df in int_data.items():
    print(f"  do({v}={val:+d}) : {len(df)} rows")

# %% [markdown]
# ## 2. Automated Exploration
#
# ### 2a. Intervention effect heatmap
#
# **Key principle:** under `do(X=v)`, every non-descendant of X keeps its
# observational distribution. We use a two-sample KS test to detect which
# variables shift — those are X's descendants.

# %%
def ks_effect(ivar, idf, obs_df):
    """KS test for each variable: intervention vs observational distribution."""
    out = {}
    for v in VARS:
        if v == ivar:
            out[v] = dict(ks=np.nan, p=np.nan, affected=None)
        else:
            ks, p = stats.ks_2samp(obs_df[v].dropna(), idf[v].dropna())
            out[v] = dict(ks=ks, p=p, affected=(p < ALPHA))
    return out


rows, row_labels = [], []
for (ivar, ival), idf in int_data.items():
    res = ks_effect(ivar, idf, obs)
    rows.append([res[v]["p"] if v != ivar else np.nan for v in VARS])
    row_labels.append(f"do({ivar}={ival:+d})")

pval_df = pd.DataFrame(rows, index=row_labels, columns=VARS)
sig_df  = (pval_df < ALPHA).astype(float)
sig_df[pval_df.isna()] = 0.5   # grey for the intervened variable itself

fig, ax = plt.subplots(figsize=(10, 3.8))
im = ax.imshow(sig_df.values, cmap="RdYlGn_r", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(len(VARS)));  ax.set_xticklabels(VARS, fontsize=13)
ax.set_yticks(range(len(row_labels))); ax.set_yticklabels(row_labels, fontsize=11)
ax.set_title(f"Intervention Effects  (red = distribution shifted, α={ALPHA})", fontsize=13)

for i, lbl in enumerate(row_labels):
    for j, v in enumerate(VARS):
        p = pval_df.iloc[i, j]
        txt = "(self)" if np.isnan(p) else (f"p<.001" if p < 0.001 else f"p={p:.3f}")
        col = "white" if (not np.isnan(p) and p < ALPHA) else "grey" if np.isnan(p) else "black"
        ax.text(j, i, txt, ha="center", va="center", fontsize=8, color=col)

plt.colorbar(im, ax=ax, ticks=[0, 0.5, 1],
             label="affected (1) / self (0.5) / unaffected (0)")
plt.tight_layout()
plt.savefig("intervention_heatmap.png", dpi=130, bbox_inches="tight")
plt.show()

# --- summary: inferred descendants per variable -------------------------
print("\nInferred descendants from intervention data (p < α):")
inferred_desc = {}
for (ivar, ival), idf in int_data.items():
    res = ks_effect(ivar, idf, obs)
    desc = {v for v, r in res.items() if r["affected"]}
    inferred_desc.setdefault(ivar, set()).update(desc)

for var in sorted(inferred_desc):
    desc     = sorted(inferred_desc[var])
    non_desc = sorted(set(VARS) - {var} - set(desc))
    print(f"  {var} →  descendants : {desc}")
    print(f"       non-descendants : {non_desc}")

# %% [markdown]
# ### 2b. Observational correlation matrix

# %%
corr = obs.corr()
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(VARS))); ax.set_xticklabels(VARS)
ax.set_yticks(range(len(VARS))); ax.set_yticklabels(VARS)
plt.colorbar(im, ax=ax, label="Pearson r")
for i in range(len(VARS)):
    for j in range(len(VARS)):
        ax.text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center", fontsize=9)
ax.set_title("Observational Correlations", fontsize=13)
plt.tight_layout()
plt.savefig("obs_correlations.png", dpi=130, bbox_inches="tight")
plt.show()

# %% [markdown]
# ### 2c. Scatter matrix for a quick eyeball

# %%
pd.plotting.scatter_matrix(obs, figsize=(10, 8), alpha=0.3, diagonal="kde")
plt.suptitle("Pairwise Scatter Matrix (observational)", y=1.01, fontsize=13)
plt.tight_layout()
plt.savefig("scatter_matrix.png", dpi=100, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 3. ★ DEFINE YOUR DAG HERE ★
#
# Format: `{ node : [list_of_parents] }`
#
# - `_H` is the hidden confounder (unobserved).  It is added automatically.
# - `h_children` = the two observable nodes that `_H` points to.
#   They **must not** have a direct edge between them (enforced in §4a).

# %%
# ============================================================
#   EDIT THIS CELL WITH YOUR HYPOTHESIS
# ============================================================

proposed_dag = {
    "A": [],            # e.g. A is a root node
    "B": ["A"],         # B ← A
    "C": ["A"],         # C ← A
    "D": ["B", "C"],    # D ← B, C
    "E": ["C"],         # E ← C
    "F": ["D", "E"],    # F ← D, E
}

# The two nodes the hidden confounder _H directly causes
# They must NOT have a direct edge between them
h_children = ["C", "E"]   # ← CHANGE THESE

# ============================================================

# --- auto-add _H ----------------------------------------------------
proposed_dag["_H"] = []
for node in h_children:
    if "_H" not in proposed_dag.get(node, []):
        proposed_dag[node] = proposed_dag.get(node, []) + ["_H"]

print("Proposed DAG  (node: parents)")
print("-" * 35)
for node, parents in sorted(proposed_dag.items()):
    print(f"  {node:4s} ← {parents if parents else '(root)'}")
print(f"\n_H children : {h_children}  →  no direct edge allowed between them")

# %% [markdown]
# ## 4. Validation Utilities  *(run as-is)*

# %%
# ── graph helpers ────────────────────────────────────────────────────────
def build_nx(dag):
    G = nx.DiGraph()
    for node, parents in dag.items():
        G.add_node(node)
        for p in parents:
            G.add_edge(p, node)
    return G


def get_descendants(dag, node, obs_only=True):
    G = build_nx(dag)
    desc = nx.descendants(G, node)
    if obs_only:
        desc = {v for v in desc if v in VARS}
    return desc


# ── statistical tests ─────────────────────────────────────────────────
def partial_corr(X, Y, Z, data):
    """Partial correlation X,Y | Z.  Returns (r, p)."""
    cols = list({X, Y, *Z} & set(data.columns))
    df = data[cols].dropna()
    if len(Z) == 0:
        return stats.pearsonr(df[X], df[Y])
    n = len(df)
    Zm = np.column_stack([df[z] for z in Z] + [np.ones(n)])
    def resid(v):
        y = df[v].values
        return y - Zm @ np.linalg.lstsq(Zm, y, rcond=None)[0]
    return stats.pearsonr(resid(X), resid(Y))


def regress_node(node, parents, data):
    """OLS: node ~ parents.  Returns (coefs_dict, r2, residuals)."""
    y = data[node].values
    if not parents:
        return {}, 0.0, y - y.mean()
    n = len(y)
    Xm = np.column_stack([data[p].values for p in parents] + [np.ones(n)])
    coef, _, _, _ = np.linalg.lstsq(Xm, y, rcond=None)
    y_hat = Xm @ coef
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    # t-stats
    mse = ss_res / max(n - len(parents) - 1, 1)
    XtX_inv = np.linalg.pinv(Xm.T @ Xm)
    se = np.sqrt(mse * np.diag(XtX_inv))
    t_vals = coef / (se + 1e-12)
    p_vals = [2 * stats.t.sf(abs(t), df=n - len(parents) - 1) for t in t_vals]
    coef_dict = {p: (coef[i], t_vals[i], p_vals[i]) for i, p in enumerate(parents)}
    return coef_dict, r2, y - y_hat


# %% [markdown]
# ## 5. Structural Checks

# %%
print("=" * 60)
print("5. STRUCTURAL CHECKS")
print("=" * 60)

G_full = build_nx(proposed_dag)

# 5a — is it a DAG?
is_acyclic = nx.is_directed_acyclic_graph(G_full)
print(f"\n[{'✓' if is_acyclic else '✗'}] No cycles (valid DAG) : {is_acyclic}")
if not is_acyclic:
    cycles = list(nx.simple_cycles(G_full))
    print(f"   Cycles found: {cycles}")

# 5b — hidden confounder constraint
if len(h_children) == 2:
    n1, n2 = h_children
    direct = (n2 in proposed_dag.get(n1, [])) or (n1 in proposed_dag.get(n2, []))
    ok = not direct
    print(f"\n[{'✓' if ok else '✗'}] No direct edge between {n1} ↔ {n2} : {ok}")
    if direct:
        print(f"   ✗  {n1} and {n2} share hidden parent _H → direct edge violates the constraint!")

# 5c — all VARS present
dag_obs_nodes = set(proposed_dag.keys()) - {"_H"}
missing = set(VARS) - dag_obs_nodes
extra   = dag_obs_nodes - set(VARS)
print(f"\n[{'✓' if not missing else '✗'}] All variables in DAG: {not bool(missing)}")
if missing: print(f"   Missing: {missing}")
if extra:   print(f"   Extra (not in data): {extra}")

# %% [markdown]
# ## 6. Intervention-Based Descendant Validation
#
# For each intervention `do(X=v)`:
# * **Proposed descendants** must show a significant distribution shift (KS p < α)
# * **Proposed non-descendants** must NOT show a shift

# %%
print("=" * 60)
print("6. INTERVENTION DESCENDANT VALIDATION")
print("=" * 60)

PASS = "✓"; FAIL = "✗"
global_pass = True

for (ivar, ival), idf in int_data.items():
    if ivar not in VARS:
        continue
    prop_desc     = get_descendants(proposed_dag, ivar)
    prop_non_desc = set(VARS) - {ivar} - prop_desc
    effects       = ks_effect(ivar, idf, obs)

    print(f"\n  do({ivar}={ival:+d})")
    print(f"  Proposed descendants    : {sorted(prop_desc)}")
    print(f"  Proposed non-descendants: {sorted(prop_non_desc)}")
    print()

    for v in sorted(prop_desc):
        p = effects[v]["p"];  affected = effects[v]["affected"]
        ok = affected
        if not ok: global_pass = False
        label = "affected ✓" if affected else "NOT affected  ← PROBLEM"
        print(f"    {PASS if ok else FAIL}  {v} [descendant]       p={p:.4f}  →  {label}")

    for v in sorted(prop_non_desc):
        p = effects[v]["p"];  affected = effects[v]["affected"]
        ok = not affected
        if not ok: global_pass = False
        label = "not affected ✓" if not affected else "AFFECTED  ← PROBLEM"
        print(f"    {PASS if ok else FAIL}  {v} [non-descendant]   p={p:.4f}  →  {label}")

print(f"\n  Overall : {'ALL PASS ✓' if global_pass else 'FAILURES DETECTED ✗'}")

# %% [markdown]
# ## 7. Conditional Independence Tests
#
# For every ordered pair (X → Y):
# * If X **is** a proposed parent of Y: `corr(X, Y | other_parents_of_Y)` should be **significant**
# * If X **is not** a parent of Y: `corr(X, Y | all_parents_of_Y)` should be **non-significant**
#
# ⚠️ Hidden confounders (_H) can create residual correlations the test can't condition on —
# flag those separately.

# %%
print("=" * 60)
print("7. CONDITIONAL INDEPENDENCE TESTS")
print("=" * 60)

ci_results = []
obs_vars = VARS  # only test observable variables

for X, Y in permutations(obs_vars, 2):
    obs_parents_Y = [p for p in proposed_dag.get(Y, []) if p in obs_vars]
    x_is_parent   = X in proposed_dag.get(Y, [])

    if x_is_parent:
        # condition on everything *except* X itself → X should still predict Y
        cond = [p for p in obs_parents_Y if p != X]
    else:
        # condition on all observable parents → X should be screened off
        cond = obs_parents_Y

    r, p = partial_corr(X, Y, cond, obs)
    actually_dep   = p < ALPHA
    should_be_dep  = x_is_parent
    consistent     = (should_be_dep == actually_dep)

    # Flag pairs that share _H as a potential source of residual correlation
    h_involved = (X in h_children and Y in h_children)

    ci_results.append(dict(X=X, Y=Y, cond=cond, r=r, p=p,
                           x_is_parent=x_is_parent, should_be_dep=should_be_dep,
                           actually_dep=actually_dep, consistent=consistent,
                           h_involved=h_involved))

ci_df = pd.DataFrame(ci_results)
failures  = ci_df[~ci_df["consistent"]]
successes = ci_df[ ci_df["consistent"]]

print(f"\nConsistent : {len(successes)}/{len(ci_df)}")
print(f"Failures   : {len(failures)}/{len(ci_df)}\n")

if not failures.empty:
    print("--- INCONSISTENCIES -----------------------------------------------")
    for _, row in failures.iterrows():
        cstr   = f" | {row['cond']}" if row['cond'] else " (marginal)"
        etype  = "proposed parent" if row['x_is_parent'] else "non-parent"
        expect = "DEPENDENT" if row['should_be_dep'] else "INDEPENDENT"
        got    = "dependent" if row['actually_dep'] else "independent"
        hflag  = "  [shared _H]" if row['h_involved'] else ""
        print(f"  ✗  {row['X']} vs {row['Y']}{cstr}   [{etype}]")
        print(f"       r={row['r']:+.3f}, p={row['p']:.4f}  |  expected {expect}, got {got}{hflag}")
    print()

print("--- ALL CONSISTENT RESULTS ----------------------------------------")
for _, row in successes.iterrows():
    cstr   = f" | {row['cond']}" if row['cond'] else ""
    status = "dep" if row['should_be_dep'] else "indep"
    print(f"  ✓  {row['X']:2s} ⊥ {row['Y']:2s}{cstr:30s}  r={row['r']:+.3f}, p={row['p']:.4f}  [{status}]")

# %% [markdown]
# ## 8. Regression Edge Strength
#
# For each node, regress on its **proposed observable parents** using OLS.
# - Parents should have **large, significant coefficients** (p < α)
# - High R² means the parents explain the node well
# - If R² is low and no parents are significant → wrong parents suspected

# %%
print("=" * 60)
print("8. REGRESSION EDGE STRENGTH")
print("=" * 60)

for node in VARS:
    obs_parents = [p for p in proposed_dag.get(node, []) if p in obs_vars]
    coef_dict, r2, resid = regress_node(node, obs_parents, obs)

    if not obs_parents:
        y = obs[node].values
        print(f"\n  {node}  (root — no observable parents)")
        print(f"    mean={y.mean():.3f},  std={y.std():.3f}")
        continue

    print(f"\n  {node} ~ {' + '.join(obs_parents)}    R²={r2:.3f}")
    for parent, (c, t, p) in coef_dict.items():
        stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "(n.s.)"
        ok_marker = "✓" if p < ALPHA else "✗"
        print(f"    {ok_marker}  {parent:>4} → {node}:  coef={c:+.4f}  t={t:.2f}  p={p:.4f}  {stars}")

    # also check residuals against non-parent obs vars — strong correlation suggests missing edge
    non_parents = [v for v in VARS if v != node and v not in obs_parents]
    suspects = []
    for np_var in non_parents:
        r_np, p_np = stats.pearsonr(resid, obs[np_var].dropna()[:len(resid)])
        if p_np < ALPHA:
            suspects.append((np_var, r_np, p_np))
    if suspects:
        print(f"    ⚠  Residuals correlated with: "
              + ", ".join(f"{v}(r={r:.2f},p={p:.3f})" for v,r,p in suspects))
        print(f"       → possible missing edge or confounder")

# %% [markdown]
# ## 9. Hidden Confounder Consistency
#
# Nodes sharing `_H` should be **correlated** in observational data.
# When we **intervene** on one (breaking the path through _H), 
# the correlation should vanish or shrink substantially.

# %%
if len(h_children) == 2:
    n1, n2 = h_children
    print("=" * 60)
    print(f"9. HIDDEN CONFOUNDER CHECK   _H → {{{n1}, {n2}}}")
    print("=" * 60)

    # -- observational correlation
    r_obs, p_obs = stats.pearsonr(obs[n1], obs[n2])
    label = "correlated ✓ (consistent with shared hidden cause)" if p_obs < ALPHA \
            else "NOT correlated ✗ (unexpected if _H → both)"
    print(f"\n  Observational  {n1} ↔ {n2}:  r={r_obs:+.3f},  p={p_obs:.4f}  →  {label}")

    # -- partial correlation conditioning on observable parents shared paths
    shared_obs_parents = (set(proposed_dag.get(n1, [])) & set(proposed_dag.get(n2, []))) - {"_H"}
    if shared_obs_parents:
        r_cond, p_cond = partial_corr(n1, n2, list(shared_obs_parents), obs)
        print(f"  Partial corr   {n1} ↔ {n2} | {list(shared_obs_parents)}:  "
              f"r={r_cond:+.3f},  p={p_cond:.4f}")
        print(f"  {'  Residual correlation persists → consistent with _H' if p_cond < ALPHA else '  Correlation vanishes — perhaps _H not needed here'}")

    # -- under each available intervention on n1 or n2
    print()
    for (ivar, ival), idf in int_data.items():
        if ivar not in h_children:
            continue
        r_int, p_int = stats.pearsonr(idf[n1], idf[n2])
        change = abs(r_obs) - abs(r_int)
        arrow  = "↓" if change > 0.05 else ("↑" if change < -0.05 else "≈")
        note = ("correlation dropped → backdoor path through _H broken ✓"
                if p_int >= ALPHA else
                "still correlated — could be a direct/downstream path too")
        print(f"  do({ivar}={ival:+d})  {n1} ↔ {n2}:  r={r_int:+.3f}  p={p_int:.4f}  "
              f"Δ|r|={change:+.3f}{arrow}  →  {note}")

# %% [markdown]
# ## 10. Summary & Graph Visualisation

# %%
print("=" * 60)
print("10. SUMMARY")
print("=" * 60)

n_ci_fail   = len(failures)
n_ci_total  = len(ci_df)
n_ci_pass   = len(successes)

# Re-run descendant check for summary count
desc_pass, desc_fail = 0, 0
for (ivar, ival), idf in int_data.items():
    if ivar not in VARS: continue
    prop_desc     = get_descendants(proposed_dag, ivar)
    prop_non_desc = set(VARS) - {ivar} - prop_desc
    effects       = ks_effect(ivar, idf, obs)
    for v in prop_desc | prop_non_desc:
        if v == ivar: continue
        affected     = effects[v]["affected"]
        should       = v in prop_desc
        if affected == should: desc_pass += 1
        else:                  desc_fail += 1

checks = [
    ("Valid DAG (no cycles)",              is_acyclic),
    (f"No direct edge {h_children[0]}↔{h_children[1]}", not direct_1_2 if len(h_children)==2 else True),
    ("All variables covered",             not bool(missing)),
    (f"Intervention descendant checks ({desc_pass}/{desc_pass+desc_fail})",
                                          desc_fail == 0),
    (f"CI tests consistent ({n_ci_pass}/{n_ci_total})",
                                          n_ci_fail == 0),
]

for label, ok in checks:
    print(f"  [{'✓' if ok else '✗'}]  {label}")

# ── Draw the DAG ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

G_vis = build_nx(proposed_dag)

try:
    pos = nx.nx_agraph.graphviz_layout(G_vis, prog="dot")
except Exception:
    pos = nx.spring_layout(G_vis, seed=42, k=2.5)

# Node colours
color_map = []
for node in G_vis.nodes():
    if node == "_H":
        color_map.append("#AAAAAA")
    elif node in h_children:
        color_map.append("#FFD166")
    else:
        color_map.append("#06B6D4")

nx.draw_networkx_nodes(G_vis, pos, node_color=color_map, node_size=2200,
                        alpha=0.92, ax=ax)
nx.draw_networkx_labels(G_vis, pos, ax=ax, font_size=13, font_weight="bold",
                         font_color="white")

# Dashed edges from _H, solid otherwise
h_edges   = [(u, v) for u, v in G_vis.edges() if u == "_H"]
obs_edges = [(u, v) for u, v in G_vis.edges() if u != "_H"]

nx.draw_networkx_edges(G_vis, pos, edgelist=obs_edges, ax=ax,
                        arrows=True, arrowsize=22,
                        edge_color="#1E293B", width=2.2,
                        connectionstyle="arc3,rad=0.08")
nx.draw_networkx_edges(G_vis, pos, edgelist=h_edges, ax=ax,
                        arrows=True, arrowsize=22,
                        edge_color="#888888", width=1.8,
                        style="dashed",
                        connectionstyle="arc3,rad=0.08")

legend_patches = [
    mpatches.Patch(color="#06B6D4", label="Observable node"),
    mpatches.Patch(color="#FFD166", label=f"_H children ({', '.join(h_children)})"),
    mpatches.Patch(color="#AAAAAA", label="Hidden confounder (_H)"),
]
ax.legend(handles=legend_patches, loc="upper right", fontsize=10,
          framealpha=0.9)
ax.set_title("Proposed DAG", fontsize=15, fontweight="bold")
ax.axis("off")
plt.tight_layout()
plt.savefig("proposed_dag.png", dpi=140, bbox_inches="tight")
plt.show()
print("\nGraph saved to proposed_dag.png")