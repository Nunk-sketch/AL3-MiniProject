"""
Demo: Example usage of graph_finder module.

This script demonstrates how to use find_valid_graphs to enumerate causal structures
for a simple test case with 4 observed variables.
"""

from graph_finder import find_valid_graphs


def main():
    """Run example graph finding with test data."""
    # Define observed variables
    observed_nodes = ["A", "B", "C", "D"]

    # Define constraints: which variable pairs MUST be connected
    # Example: A and B are not d-separated (must have a path)
    #          C and D are not d-separated (must have a path)
    connections = [("A", "B"), ("C", "D")]

    print("=" * 60)
    print("Graph Finder Demo")
    print("=" * 60)
    print(f"Observed variables: {observed_nodes}")
    print(f"Connection constraints: {connections}")
    print()

    # Find all valid DAG configurations
    # When h_confounds=None, the function searches all possible pairs
    results = find_valid_graphs(observed_nodes, connections)

    print()
    print("=" * 60)
    print(f"Found {len(results)} valid DAG configurations.")
    print("=" * 60)

    if results:
        print("\nExample Graph (first configuration):")
        for edge in results[0]:
            print(f"  {edge[0]} → {edge[1]}")

    # Example with specific h_confounds
    print("\n" + "=" * 60)
    print("Testing with specific confounding pair: H confounds A and C")
    print("=" * 60)

    results_specific = find_valid_graphs(
        observed_nodes, connections, h_confounds=("A", "C")
    )

    print(f"Found {len(results_specific)} valid DAG configurations.")


if __name__ == "__main__":
    main()
