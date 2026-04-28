"""
Graph Finder: Enumerate valid Directed Acyclic Graphs (DAGs) consistent with data constraints.

This module finds all valid causal graph structures given:
- Observed variables (A, B, C, etc.)
- A single hidden confounder (_H) that affects exactly two variables
- Constraints that certain variable pairs must be connected (reachable)

The algorithm uses brute-force enumeration of all possible edge configurations,
checking each for DAG validity and constraint satisfaction.
"""

from typing import List, Tuple, Optional
import networkx as nx
from itertools import combinations


def find_valid_graphs(
    observed_nodes: List[str],
    connections: List[Tuple[str, str]],
    h_confounds: Optional[Tuple[str, str]] = None,
) -> List[List[Tuple[str, str]]]:
    """
    Enumerate all valid DAGs consistent with observed constraints and a hidden confounder.

    This function generates all possible causal graph structures where:
    - A hidden variable _H confounds (has causal arrows to) exactly two observed variables
    - No direct edge exists between the two variables confounded by _H
    - All specified connections (pairs that must have some path between them) are reachable

    Args:
        observed_nodes: List of observed variable names, e.g., ['A', 'B', 'C', 'D']
        connections: List of (u, v) tuples. Each represents a pair of variables that
                     MUST have a path between them (in undirected sense, not d-separated).
                     Example: [('A', 'B'), ('C', 'D')] means A-B must be connected and
                     C-D must be connected.
        h_confounds: Optional tuple (var1, var2) specifying which two variables _H confounds.
                     If None, the function iterates through all possible pairs.
                     Default: None (search all pairs)

    Returns:
        List of valid graphs, where each graph is represented as a list of directed edges.
        Each edge is a tuple (source, target), e.g., [('_H', 'A'), ('_H', 'B'), ('A', 'C')].

    Raises:
        ValueError: If h_confounds is provided but doesn't contain exactly 2 variables,
                    or if any node in h_confounds is not in observed_nodes.

    Example:
        >>> observed = ['A', 'B', 'C', 'D']
        >>> must_connect = [('A', 'B'), ('C', 'D')]
        >>> graphs = find_valid_graphs(observed, must_connect)
        >>> len(graphs)  # Number of valid DAG configurations
        1024  # For 4 nodes, typically in the hundreds to thousands range

    Complexity:
        - Time: O(2^m * n^2) where m = number of potential edges, n = number of nodes
        - For 4 observed nodes (5 total): ~6144 iterations (6 H-pairs × 2^10 combinations)
        - For 6 observed nodes (7 total): significantly larger (~30k+ iterations)
    """
    # Validate h_confounds if provided
    if h_confounds is not None:
        if len(h_confounds) != 2:
            raise ValueError(f"h_confounds must be a tuple of 2 variables, got {len(h_confounds)}")
        if not all(node in observed_nodes for node in h_confounds):
            raise ValueError(f"h_confounds variables must be in observed_nodes")

    # Build node mappings
    all_nodes = observed_nodes + ["_H"]
    n = len(all_nodes)
    node_to_idx = {node: i for i, node in enumerate(all_nodes)}
    idx_to_node = {i: node for i, node in enumerate(all_nodes)}
    h_idx = node_to_idx["_H"]

    # Determine which pairs _H could confound
    h_target_pairs = (
        [h_confounds] if h_confounds else list(combinations(observed_nodes, 2))
    )

    valid_graphs = []

    for target1, target2 in h_target_pairs:
        t1, t2 = node_to_idx[target1], node_to_idx[target2]

        # Define potential edges between observed nodes
        # Rule: No direct link between the two variables confounded by _H
        potential_edges = []
        for u, v in combinations(observed_nodes, 2):
            u_idx, v_idx = node_to_idx[u], node_to_idx[v]
            if (u_idx == t1 and v_idx == t2) or (u_idx == t2 and v_idx == t1):
                continue  # Forbidden edge
            potential_edges.append((u_idx, v_idx))
            potential_edges.append((v_idx, u_idx))

        # Base edges: H must point to both confounded variables
        base_edges = [(h_idx, t1), (h_idx, t2)]

        print(f"Checking H confounding {target1} and {target2}...")

        # Enumerate all subsets of potential_edges using bit masking
        num_potentials = len(potential_edges)
        for r in range(1 << num_potentials):
            current_edges = base_edges.copy()
            for i in range(num_potentials):
                if (r >> i) & 1:
                    current_edges.append(potential_edges[i])

            # 1. Check if it's a DAG
            G = nx.DiGraph()
            G.add_nodes_from(range(n))
            G.add_edges_from(current_edges)

            if nx.is_directed_acyclic_graph(G):
                # 2. Check Reachability (The "Not Separated" constraint)
                # Convert to undirected to check if variables can be connected via any path
                # (ignoring edge direction, since confounding bypasses directionality)
                undirected_G = G.to_undirected()
                if all(
                    nx.has_path(undirected_G, node_to_idx[u], node_to_idx[v])
                    for u, v in connections
                ):
                    # Convert back to named edges for storage
                    named_edges = [
                        (idx_to_node[u], idx_to_node[v]) for u, v in current_edges
                    ]
                    valid_graphs.append(named_edges)

    return valid_graphs
