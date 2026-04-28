import numpy as np
import networkx as nx
from itertools import combinations, product

def find_valid_graphs(observed_nodes, connections, h_confounds=None):
    """
    observed_nodes: List of strings e.g. ['A', 'B', 'C', 'D']
    connections: List of tuples (u, v) that MUST have a path between them 
                 (not necessarily a direct edge, just not d-separated)
    h_confounds: Optional tuple of 2 nodes H is known to confound. 
                 If None, we iterate through all possible pairs.
    """
    all_nodes = observed_nodes + ['_H']
    n = len(all_nodes)
    node_to_idx = {node: i for i, node in enumerate(all_nodes)}
    idx_to_node = {i: node for i, node in enumerate(all_nodes)}
    h_idx = node_to_idx['_H']
    
    # Possible pairs H could confound
    h_target_pairs = [h_confounds] if h_confounds else list(combinations(observed_nodes, 2))
    
    valid_graphs = []

    for target1, target2 in h_target_pairs:
        t1, t2 = node_to_idx[target1], node_to_idx[target2]
        
        # Define potential edges between observed nodes
        # Rule: No direct link between the two variables confounded by _H
        potential_edges = []
        for u, v in combinations(observed_nodes, 2):
            u_idx, v_idx = node_to_idx[u], node_to_idx[v]
            if (u_idx == t1 and v_idx == t2) or (u_idx == t2 and v_idx == t1):
                continue # Forbidden edge
            potential_edges.append((u_idx, v_idx))
            potential_edges.append((v_idx, u_idx))

        # We must have H -> target1 and H -> target2
        base_edges = [(h_idx, t1), (h_idx, t2)]
        
        # Power set of potential edges is too big for 7 nodes.
        # Instead, we use a recursive approach to build the DAG
        print(f"Checking H confounding {target1} and {target2}...")
        
        # For 5 nodes (A,B,C,D,H), observed edges = (4*3)-2 = 10 edges. 
        # 2^10 = 1024 combinations per H-pair. Total 6 * 1024 = 6144. Very fast.
        # For 7 nodes (A,B,C,D,E,F,H), it's significantly more.
        
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
                # 2. Check Reachability (The "Not Separated" info)
                # This checks if there is ANY path (directed or via H) between the nodes
                undirected_G = G.to_undirected()
                if all(nx.has_path(undirected_G, node_to_idx[u], node_to_idx[v]) for u, v in connections):
                    # Convert back to named tuples for storage
                    named_edges = [(idx_to_node[u], idx_to_node[v]) for u, v in current_edges]
                    valid_graphs.append(named_edges)

    return valid_graphs

# --- SETUP DATA ---
OBS_NODES = ['A', 'B', 'C', 'D'] 
# Example: We know A and B are connected, and C and D are connected
CONNS = [('A', 'B'), ('C', 'D')]

results = find_valid_graphs(OBS_NODES, CONNS)

print(f"\nFound {len(results)} valid DAG configurations.")
if results:
    print("Example Graph:", results[0])