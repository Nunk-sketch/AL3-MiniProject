"""
Unit tests for the graph_finder module.

Tests verify:
1. DAG validity checking
2. Reachability constraints
3. Confounding pair enumeration
4. Scalability and performance
5. Error handling
"""

import unittest
import time
from graph_finder import find_valid_graphs


class TestGraphFinder(unittest.TestCase):
    """Test suite for find_valid_graphs function."""

    def test_simple_4_node_case(self):
        """Test basic 4-node (5 with _H) case."""
        observed = ["A", "B", "C", "D"]
        connections = [("A", "B"), ("C", "D")]
        
        graphs = find_valid_graphs(observed, connections)
        
        # Should find valid configurations
        self.assertGreater(len(graphs), 0)
        
        # Each graph should be a list of edges
        for graph in graphs:
            self.assertIsInstance(graph, list)
            for edge in graph:
                self.assertEqual(len(edge), 2)
                self.assertIsInstance(edge, tuple)
    
    def test_specific_confound_pair(self):
        """Test with a specific H confounding pair."""
        observed = ["A", "B", "C", "D"]
        connections = [("A", "B")]
        
        # Test with specific confounding pair
        graphs_specific = find_valid_graphs(
            observed, connections, h_confounds=("A", "C")
        )
        
        # Should find valid configurations
        self.assertGreater(len(graphs_specific), 0)
        
        # All graphs should have edges from _H to A and C
        for graph in graphs_specific:
            edges = set(graph)
            self.assertIn(("_H", "A"), edges)
            self.assertIn(("_H", "C"), edges)
    
    def test_no_direct_edge_between_confounded_vars(self):
        """Verify no direct edge exists between confounded variables."""
        observed = ["A", "B", "C", "D"]
        connections = [("A", "B"), ("C", "D")]
        
        graphs = find_valid_graphs(observed, connections, h_confounds=("A", "B"))
        
        # Check that no graph has a direct A -> B or B -> A edge
        for graph in graphs:
            edges = set(graph)
            self.assertNotIn(("A", "B"), edges)
            self.assertNotIn(("B", "A"), edges)
    
    def test_reachability_constraint(self):
        """Test that reachability constraints are enforced."""
        observed = ["A", "B", "C", "D"]
        connections = [("A", "B")]  # A and B must be connected
        
        graphs = find_valid_graphs(observed, connections)
        
        # Verify all graphs satisfy reachability
        import networkx as nx
        for graph in graphs:
            G = nx.DiGraph()
            G.add_nodes_from(observed + ["_H"])
            G.add_edges_from(graph)
            G_undirected = G.to_undirected()
            
            # A and B should have a path
            self.assertTrue(nx.has_path(G_undirected, "A", "B"))
    
    def test_invalid_confound_pair_raises_error(self):
        """Test that invalid h_confounds raises ValueError."""
        observed = ["A", "B", "C", "D"]
        connections = []
        
        # Invalid: confound pair with only 1 element
        with self.assertRaises(ValueError):
            find_valid_graphs(observed, connections, h_confounds=("A",))
        
        # Invalid: confound pair with 3 elements
        with self.assertRaises(ValueError):
            find_valid_graphs(observed, connections, h_confounds=("A", "B", "C"))
        
        # Invalid: variable not in observed_nodes
        with self.assertRaises(ValueError):
            find_valid_graphs(observed, connections, h_confounds=("A", "X"))
    
    def test_h_node_always_present(self):
        """Test that _H node is always in output graphs."""
        observed = ["A", "B", "C", "D"]
        connections = [("A", "B")]
        
        graphs = find_valid_graphs(observed, connections)
        
        for graph in graphs:
            edge_nodes = set()
            for u, v in graph:
                edge_nodes.add(u)
                edge_nodes.add(v)
            
            # _H should be present
            self.assertIn("_H", edge_nodes)
    
    def test_all_graphs_are_dags(self):
        """Test that all returned graphs are DAGs."""
        import networkx as nx
        
        observed = ["A", "B", "C", "D"]
        connections = [("A", "B"), ("C", "D")]
        
        graphs = find_valid_graphs(observed, connections)
        
        for graph in graphs:
            G = nx.DiGraph()
            G.add_nodes_from(observed + ["_H"])
            G.add_edges_from(graph)
            
            # Must be a DAG (no cycles)
            self.assertTrue(nx.is_directed_acyclic_graph(G))
    
    def test_3_node_case(self):
        """Test smaller case with 3 observed nodes."""
        observed = ["A", "B", "C"]
        connections = [("A", "B")]
        
        graphs = find_valid_graphs(observed, connections)
        
        # Should find valid configurations
        self.assertGreater(len(graphs), 0)
    
    def test_empty_connections(self):
        """Test with no connection constraints."""
        observed = ["A", "B", "C", "D"]
        connections = []
        
        graphs = find_valid_graphs(observed, connections)
        
        # Should find many valid DAGs with no constraints
        self.assertGreater(len(graphs), 100)
    
    def test_performance_4_nodes(self):
        """Test performance on 4 observed nodes."""
        observed = ["A", "B", "C", "D"]
        connections = [("A", "B")]
        
        start = time.time()
        graphs = find_valid_graphs(observed, connections)
        elapsed = time.time() - start
        
        # Should complete in < 1 second for 4 nodes
        self.assertLess(elapsed, 1.0, f"4-node case took {elapsed:.2f}s")
        
        if True:  # Set to True to see timing
            print(f"\n4-node case: {len(graphs)} graphs in {elapsed:.3f}s")
    
    def test_performance_6_nodes(self):
        """Test performance on 6 observed nodes."""
        observed = ["A", "B", "C", "D", "E", "F"]
        connections = [("A", "B"), ("C", "D")]
        
        start = time.time()
        graphs = find_valid_graphs(observed, connections)
        elapsed = time.time() - start
        
        # Should complete in reasonable time (allow up to 30s for slower machines)
        self.assertLess(elapsed, 30.0, f"6-node case took {elapsed:.2f}s")
        
        if True:  # Set to True to see timing
            print(f"\n6-node case: {len(graphs)} graphs in {elapsed:.3f}s")
    
    def test_specific_confound_reduces_search_space(self):
        """Test that specifying h_confounds reduces the number of results."""
        observed = ["A", "B", "C", "D"]
        connections = []
        
        # Search all confounding pairs
        graphs_all = find_valid_graphs(observed, connections, h_confounds=None)
        
        # Search specific confounding pair
        graphs_specific = find_valid_graphs(
            observed, connections, h_confounds=("A", "B")
        )
        
        # Specific should be a subset (smaller)
        self.assertLess(len(graphs_specific), len(graphs_all))
        
        if True:
            print(f"\nAll pairs: {len(graphs_all)} graphs")
            print(f"A,B pair: {len(graphs_specific)} graphs")


class TestGraphProperties(unittest.TestCase):
    """Test properties of generated graphs."""
    
    def test_h_edges_to_confounded_vars(self):
        """Test that _H has edges to exactly its confounded variables."""
        import networkx as nx
        
        observed = ["A", "B", "C", "D"]
        connections = []
        
        graphs = find_valid_graphs(
            observed, connections, h_confounds=("B", "D")
        )
        
        for graph in graphs:
            h_targets = set()
            for u, v in graph:
                if u == "_H":
                    h_targets.add(v)
            
            # _H should point to B and D
            self.assertIn("B", h_targets)
            self.assertIn("D", h_targets)
    
    def test_no_self_loops(self):
        """Test that no graph has self-loops."""
        observed = ["A", "B", "C", "D"]
        connections = [("A", "B")]
        
        graphs = find_valid_graphs(observed, connections)
        
        for graph in graphs:
            for u, v in graph:
                self.assertNotEqual(u, v, "Found self-loop")
    
    def test_edge_count_reasonable(self):
        """Test that edge counts are within reasonable bounds."""
        observed = ["A", "B", "C", "D"]
        connections = []
        
        graphs = find_valid_graphs(observed, connections)
        
        for graph in graphs:
            # Minimum: 2 edges (H -> confounded1, H -> confounded2)
            self.assertGreaterEqual(len(graph), 2)
            
            # Maximum: 2 + C(4,2)*2 = 2 + 6*2 = 14 edges
            # (all possible directed edges between 4 nodes + 2 H edges)
            self.assertLessEqual(len(graph), 14)


def run_tests_verbose():
    """Run tests with verbose output."""
    suite = unittest.TestLoader().loadTestsFromModule(__import__(__name__))
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)


if __name__ == "__main__":
    # Run with verbose output
    run_tests_verbose()
