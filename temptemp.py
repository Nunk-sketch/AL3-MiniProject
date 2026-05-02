import os
import re
from typing import Dict, List, Set, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
BASELINE_FILE = "data_2358.csv"

EFFECT_THRESHOLD = 0.15
HIDDEN_NODE_NAMES = {"H", "_H"}


GRAPH_1_EDGES = [
	("B", "A"),
	("B", "C"),
	("D", "A"),
	("E", "A"),
	("E", "B"),
	("E", "D"),
	("E", "F"),
	("F", "D"),
	("H", "C"),
	("H", "F"),
]

GRAPH_2_EDGES = [
	("A", "C"),
	("B", "A"),
	("B", "E"),
	("C", "F"),
	("D", "A"),
	("E", "D"),
	("E", "F"),
	("_H", "F"),
	("_H", "D"),
	("R", "D"),
]


def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
	return df.drop(columns=["Unnamed: 0"], errors="ignore")


def load_baseline() -> pd.DataFrame:
	baseline_path = os.path.join(DATA_DIR, BASELINE_FILE)
	if not os.path.exists(baseline_path):
		raise FileNotFoundError(f"Baseline file not found: {baseline_path}")
	return _clean_df(pd.read_csv(baseline_path))


def find_intervention_files() -> Dict[str, Dict[str, str]]:
	pattern = re.compile(r"^(-?)2([A-Za-z])\.csv$")
	intervention_map: Dict[str, Dict[str, str]] = {}

	for file_name in os.listdir(DATA_DIR):
		m = pattern.match(file_name)
		if not m:
			continue

		sign, node = m.groups()
		node = node.upper()
		kind = "neg" if sign == "-" else "pos"
		intervention_map.setdefault(node, {})[kind] = os.path.join(DATA_DIR, file_name)

	return intervention_map


def compute_intervention_effect_matrix(
	baseline_df: pd.DataFrame,
	intervention_files: Dict[str, Dict[str, str]],
) -> pd.DataFrame:
	observed_nodes = list(baseline_df.columns)
	baseline_std = baseline_df.std(ddof=0).replace(0, np.nan)

	rows: Dict[str, pd.Series] = {}

	for source_node in sorted(intervention_files.keys()):
		file_info = intervention_files[source_node]

		if "pos" in file_info and "neg" in file_info:
			pos_df = _clean_df(pd.read_csv(file_info["pos"]))
			neg_df = _clean_df(pd.read_csv(file_info["neg"]))
			delta = pos_df[observed_nodes].mean() - neg_df[observed_nodes].mean()
		elif "pos" in file_info:
			pos_df = _clean_df(pd.read_csv(file_info["pos"]))
			delta = pos_df[observed_nodes].mean() - baseline_df[observed_nodes].mean()
		elif "neg" in file_info:
			neg_df = _clean_df(pd.read_csv(file_info["neg"]))
			delta = baseline_df[observed_nodes].mean() - neg_df[observed_nodes].mean()
		else:
			continue

		effect = (delta / baseline_std).abs().replace([np.inf, -np.inf], np.nan).fillna(0.0)
		rows[source_node] = effect

	if not rows:
		raise ValueError("No intervention files found in data directory.")

	matrix = pd.DataFrame.from_dict(rows, orient="index")
	matrix = matrix.reindex(sorted(matrix.index))
	matrix = matrix[observed_nodes]
	return matrix


def build_graph(edges: List[Tuple[str, str]]) -> nx.DiGraph:
	g = nx.DiGraph()
	g.add_edges_from(edges)
	return g


def reachable_observed_targets(
	graph: nx.DiGraph,
	source: str,
	observed_nodes: Set[str],
) -> Set[str]:
	if source not in graph:
		return set()

	descendants = nx.descendants(graph, source)
	return {node for node in descendants if node in observed_nodes}


def score_candidate_graph(
	graph_name: str,
	graph: nx.DiGraph,
	effect_matrix: pd.DataFrame,
	observed_nodes: Set[str],
	effect_threshold: float,
) -> Dict[str, object]:
	intervened_sources = list(effect_matrix.index)
	# Evaluate only nodes where interventions exist to keep scores comparable.
	evaluated_targets = set(intervened_sources)

	unknown_non_hidden_nodes = sorted(
		node for node in graph.nodes if (node not in observed_nodes and node not in HIDDEN_NODE_NAMES)
	)
	is_dag = nx.is_directed_acyclic_graph(graph)

	tp = fp = fn = tn = 0

	for source in intervened_sources:
		predicted_targets = reachable_observed_targets(graph, source, observed_nodes)

		for target in sorted(evaluated_targets):
			observed_effect = float(effect_matrix.loc[source, target]) >= effect_threshold
			predicted_effect = target in predicted_targets

			if predicted_effect and observed_effect:
				tp += 1
			elif predicted_effect and not observed_effect:
				fp += 1
			elif (not predicted_effect) and observed_effect:
				fn += 1
			else:
				tn += 1

	precision = tp / (tp + fp) if (tp + fp) else 0.0
	recall = tp / (tp + fn) if (tp + fn) else 0.0
	f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
	tnr = tn / (tn + fp) if (tn + fp) else 0.0
	balanced_acc = 0.5 * (recall + tnr)

	return {
		"name": graph_name,
		"is_dag": is_dag,
		"unknown_non_hidden_nodes": unknown_non_hidden_nodes,
		"tp": tp,
		"fp": fp,
		"fn": fn,
		"tn": tn,
		"precision": precision,
		"recall": recall,
		"f1": f1,
		"balanced_acc": balanced_acc,
	}


def print_effect_summary(effect_matrix: pd.DataFrame, threshold: float) -> None:
	print("Intervention-effect matrix (absolute standardized mean shifts):")
	print(effect_matrix.round(3))
	print(f"Empirically strong effects (threshold >= {threshold:.2f}, excluding self-effects):")

	found_any = False
	for source in effect_matrix.index:
		for target in effect_matrix.columns:
			if source != target and float(effect_matrix.loc[source, target]) >= threshold:
				found_any = True
				print(f"  {source} -> {target}")

	if not found_any:
		print("  (none)")


def print_graph_result(result: Dict[str, object]) -> None:
	print(result["name"])
	print("-" * len(result["name"]))
	print(f"Is DAG: {result['is_dag']}")

	unknown = result["unknown_non_hidden_nodes"]
	if unknown:
		print(f"Unknown/unobserved non-hidden nodes in graph: {', '.join(unknown)}")
	else:
		print("Unknown/unobserved non-hidden nodes in graph: None")

	print(
		f"Confusion counts: TP={result['tp']}, FP={result['fp']}, "
		f"FN={result['fn']}, TN={result['tn']}"
	)
	print(
		"Scores: "
		f"precision={result['precision']:.3f}, "
		f"recall={result['recall']:.3f}, "
		f"f1={result['f1']:.3f}, "
		f"balanced_acc={result['balanced_acc']:.3f}"
	)

	if not result["is_dag"]:
		print("Note: This candidate is cyclic, so it is invalid as a DAG.")


def extract_strong_edges_from_effects(
	effect_matrix: pd.DataFrame,
	threshold: float,
) -> List[Tuple[str, str]]:
	"""Extract edges with strong intervention effects, excluding self-effects."""
	edges = []
	for source in effect_matrix.index:
		for target in effect_matrix.columns:
			if source != target and float(effect_matrix.loc[source, target]) >= threshold:
				edges.append((source, target))
	return edges


def suggest_candidate_graphs(
	effect_matrix: pd.DataFrame,
	observed_nodes: Set[str],
	threshold: float,
) -> List[Tuple[str, List[Tuple[str, str]]]]:
	"""Suggest minimal DAGs based on observed effects."""
	strong_edges = extract_strong_edges_from_effects(effect_matrix, threshold)

	candidates = []

	# Candidate 1: All strong edges (if acyclic)
	g1 = nx.DiGraph()
	g1.add_nodes_from(observed_nodes)
	g1.add_edges_from(strong_edges)
	if nx.is_directed_acyclic_graph(g1):
		candidates.append(("All strong effects", strong_edges))

	# Candidate 2: Only D->E (not E->D) to avoid cycle
	edges_no_feedback = [e for e in strong_edges if not (e == ("E", "D"))]
	g2 = nx.DiGraph()
	g2.add_nodes_from(observed_nodes)
	g2.add_edges_from(edges_no_feedback)
	if nx.is_directed_acyclic_graph(g2):
		candidates.append(("D->E only (no E->D feedback)", edges_no_feedback))

	# Candidate 3: Only E->D (not D->E) to avoid cycle
	edges_reverse = [e for e in strong_edges if not (e == ("D", "E"))]
	g3 = nx.DiGraph()
	g3.add_nodes_from(observed_nodes)
	g3.add_edges_from(edges_reverse)
	if nx.is_directed_acyclic_graph(g3):
		candidates.append(("E->D only (no D->E feedback)", edges_reverse))

	# Candidate 4: Strong effects plus hidden confounder H between D and E
	edges_with_h = [e for e in strong_edges if not (e == ("D", "E") or e == ("E", "D"))]
	edges_with_h.extend([("H", "D"), ("H", "E")])
	g4 = nx.DiGraph()
	g4.add_nodes_from(observed_nodes | {"H"})
	g4.add_edges_from(edges_with_h)
	if nx.is_directed_acyclic_graph(g4):
		candidates.append(("Strong effects + H confounds D,E", edges_with_h))

	return candidates


def visualize_graph(graph: nx.DiGraph, title: str, filename: str) -> None:
	"""Create and save a visualization of the causal graph."""
	fig, ax = plt.subplots(figsize=(12, 8))

	# Use spring layout for better visualization
	pos = nx.spring_layout(graph, k=2, iterations=50, seed=42)

	# Draw nodes with different colors for hidden vs observed
	node_colors = []
	for node in graph.nodes():
		if node in HIDDEN_NODE_NAMES:
			node_colors.append("#FF6B6B")  # Red for hidden
		else:
			node_colors.append("#4ECDC4")  # Teal for observed

	nx.draw_networkx_nodes(
		graph, pos, node_color=node_colors, node_size=2500, ax=ax
	)
	nx.draw_networkx_labels(graph, pos, font_size=16, font_weight="bold", ax=ax)

	# Draw edges with arrows
	nx.draw_networkx_edges(
		graph, pos, edge_color="black", arrows=True, 
		arrowsize=25, arrowstyle="-|>", width=2, ax=ax,
		connectionstyle="arc3,rad=0.1"
	)

	ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
	ax.axis("off")
	plt.tight_layout()
	plt.savefig(filename, dpi=150, bbox_inches="tight")
	print(f"  Saved: {filename}")
	plt.close()


def main() -> None:
	baseline_df = load_baseline()
	observed_nodes = set(baseline_df.columns.tolist())

	intervention_files = find_intervention_files()
	effect_matrix = compute_intervention_effect_matrix(baseline_df, intervention_files)

	print(f"Observed nodes from baseline: {sorted(observed_nodes)}")
	print(f"Intervened source nodes found: {list(effect_matrix.index)}")
	print_effect_summary(effect_matrix, EFFECT_THRESHOLD)

	# Test provided graphs
	graph_1 = build_graph(GRAPH_1_EDGES)
	graph_2 = build_graph(GRAPH_2_EDGES)

	result_1 = score_candidate_graph(
		graph_name="Graph 1",
		graph=graph_1,
		effect_matrix=effect_matrix,
		observed_nodes=observed_nodes,
		effect_threshold=EFFECT_THRESHOLD,
	)
	result_2 = score_candidate_graph(
		graph_name="Graph 2",
		graph=graph_2,
		effect_matrix=effect_matrix,
		observed_nodes=observed_nodes,
		effect_threshold=EFFECT_THRESHOLD,
	)

	print_graph_result(result_1)
	print_graph_result(result_2)

	valid_results = [r for r in [result_1, result_2] if r["is_dag"]]
	if valid_results:
		best = max(valid_results, key=lambda r: (r["f1"], r["balanced_acc"]))
	else:
		best = max([result_1, result_2], key=lambda r: (r["f1"], r["balanced_acc"]))

	print(f"Best-supported graph (by intervention consistency): {best['name']}")


if __name__ == "__main__":
	main()
