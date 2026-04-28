# AL3-MiniProject
**02463 - Active Machine Learning and Agency**  
Causality and Graph Deduction

## Project Overview

This project implements a **causal graph inference system** that deduces the structure of causal relationships from observational data, accounting for hidden confounders. The system:

1. **Analyzes** data to identify statistical relationships (correlations and mutual information)
2. **Extracts** constraints (which variables must be connected)
3. **Enumerates** all valid causal structures (DAGs) consistent with constraints and known confounding
4. **Represents** a foundation for active learning to disambiguate between candidate graphs

## Key Assumptions

- **No direct edge** between the two variables confounded by the hidden confounder _H
- **Single hidden confounder** (_H) that affects exactly two observed variables
- **Observational data** from which we extract statistical associations
- **Interventions** can be performed to test causal hypotheses (future phase)

## Project Structure

```
├── main.py                      # Main workflow orchestrator
├── config.py                    # Configuration and parameters
├── graph_finder.py              # DAG enumeration engine
├── constraints_extractor.py     # Constraint extraction from analysis
├── demo_find_graphs.py          # Example usage of graph_finder
├── test_graph_finder.py         # Unit tests
├── Lin_correlation.py           # Correlation analysis (Pearson)
├── correlation_non_lin.py       # Nonlinear correlation examples
├── mutual_info.py               # Mutual information analysis
├── stats.py                     # Statistical visualizations
├── heatmap.py                   # Correlation heatmap visualization
├── data/
│   ├── data_1504.csv            # Test data (1504 samples, 4 variables)
│   └── find.py                  # [Deprecated] Use graph_finder.py instead
└── outputs/                     # Generated results
    ├── valid_graphs.json        # All valid DAG structures found
    ├── extracted_constraints.json
    └── correlation_analysis.csv
```

## Workflow

```
┌─────────────────────────────────────────────────────────────┐
│                    CAUSAL INFERENCE PIPELINE                 │
└─────────────────────────────────────────────────────────────┘

1. LOAD DATA (main.py)
   └─> CSV file with observed variables A, B, C, D, ...

2. ANALYZE (main.py + analysis modules)
   ├─> Lin_correlation.py: Pearson correlations
   ├─> mutual_info.py: Mutual information (detects nonlinearities)
   └─> Extract strong associations (threshold-based)

3. EXTRACT CONSTRAINTS (constraints_extractor.py)
   ├─> Correlation threshold: |r| ≥ 0.1
   ├─> MI threshold: MI ≥ 0.05
   └─> Result: List of (var1, var2) pairs that MUST be connected

4. ENUMERATE DAGs (graph_finder.py)
   ├─> Fixed: _H → confounded_var1, _H → confounded_var2
   ├─> Forbidden: Direct edge between confounded vars
   ├─> Optional edges: All other combinations
   └─> Valid DAGs: Those satisfying all constraints

5. OUTPUT & VISUALIZATION
   ├─> JSON files with valid graph structures
   ├─> Statistics on number of candidates
   └─> Prepare for active learning phase
```

## Quick Start

### Run the full pipeline:
```bash
python main.py
```

This will:
- Load `data/data_1504.csv`
- Compute correlations and mutual information
- Extract constraints
- Find all valid DAGs
- Save results to `outputs/`

### Run tests:
```bash
python -m pytest test_graph_finder.py -v
# or
python test_graph_finder.py
```

### Run the demo:
```bash
python demo_find_graphs.py
```

## Configuration

Edit `config.py` to customize:

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `OBSERVED_NODES` | `['A','B','C','D']` | Variable names to analyze |
| `CORRELATION_THRESHOLD` | `0.1` | Min \|correlation\| for constraint |
| `MI_THRESHOLD` | `0.05` | Min mutual information for constraint |
| `HIDDEN_CONFOUND_PAIRS` | `None` | Specific pair _H confounds (None = search all) |
| `TEST_DATA_PATH` | `data/data_1504.csv` | Input CSV file |

### Example: Analyze with 6 variables
```python
# In config.py:
OBSERVED_NODES = ["A", "B", "C", "D", "E", "F"]
TOTAL_NODES = 7
```

### Example: Higher correlation threshold
```python
# In config.py:
CORRELATION_THRESHOLD = 0.3  # Only very strong correlations
```

## Core Modules

### `graph_finder.py`
Enumerates valid Directed Acyclic Graphs (DAGs) via brute-force power-set enumeration.

**Function:** `find_valid_graphs(observed_nodes, connections, h_confounds=None)`

**Key properties:**
- Returns all DAGs where:
  - No cycles exist (DAG property)
  - All specified connections are reachable (in undirected sense)
  - No direct edge between _H's confounded variables
  - _H points to exactly its two confounded variables

**Complexity:**
- 4 variables: ~6,144 iterations (fast, <1s)
- 6 variables: ~30,000+ iterations (slower, <30s)

### `constraints_extractor.py`
Derives connection constraints from statistical analysis.

**Functions:**
- `extract_from_correlations()`: Identifies correlated variable pairs
- `extract_from_mutual_information()`: Identifies dependent variable pairs
- `combine_constraints()`: Merges results and deduplicates

### `main.py`
Orchestrates the full pipeline: data → analysis → constraints → graph enumeration → results.

## Example Output

```
======================================================================
AL3-MiniProject: Causal Graph Inference
======================================================================
✓ Loaded data from data/data_1504.csv
  Shape: (1504, 4)
  Columns: ['A', 'B', 'C', 'D']

✓ Correlation Analysis (6 pairs):
  Threshold: 0.1
  Top 5 correlations by magnitude:
    A -- D: 0.7834
    A -- C: 0.3421
    ...

✓ Constraint Extraction:
  Correlation constraints: 3
  Mutual information constraints: 2
  Combined (deduplicated): 4
  Connection pairs: [('A', 'B'), ('A', 'D'), ('C', 'D'), ('B', 'C')]

✓ Finding Valid DAG Structures:
  Observed variables: ['A', 'B', 'C', 'D']
  Total nodes (including _H): 5
  Constraints to satisfy: 4
  Searching all possible confound pairs...

Checking H confounding A and B...
Checking H confounding A and C...
[...progress...]

✓ Found 1247 valid DAG configurations

======================================================================
Pipeline completed successfully!
  Total valid DAGs found: 1247
  Connection constraints: 4
  Output directory: outputs/
======================================================================
```

## Data Format

Input CSV with numeric columns (variable names are automatically detected):

```
,A,B,C,D
0,1.809,-0.108,-0.624,2.917
1,2.616,-0.559,1.550,5.175
...
```

## Development Notes

### Reachability vs D-Separation
The current implementation uses **reachability** (any path in undirected graph) rather than formal **d-separation**.
- **Simpler** and sufficient for MVP
- **Conservative** (may exclude some valid structures)
- Can be enhanced with formal d-separation if needed

### Performance Considerations
For 6+ observed variables, the search space grows exponentially:
- 4 variables: ~1,000 DAGs per confound pair
- 6 variables: ~100,000+ DAGs per confound pair

If performance becomes a bottleneck, consider:
- Constraint propagation to prune search space
- SAT solver for formal constraint satisfaction
- Parallel enumeration of confound pairs

## Future Enhancements

1. **Intervention Planning** (Active Learning)
   - Suggest experiments to disambiguate between candidate graphs
   - Optimize for cost (samples, experiments, incorrect guesses)

2. **D-Separation Checking**
   - Replace reachability with formal d-separation
   - More accurate conditional independence testing

3. **Visualization**
   - Interactive graph exploration
   - Highlight high-confidence vs ambiguous structures

4. **Multiple Hidden Confounders**
   - Support for 2+ independent hidden confounders
   - More complex constraint propagation

## References

- Course: 02463 - Active Machine Learning and Agency, DTU
- Assumption: No direct link between confounded variables (Pearl's causal models)
- Data structure: 4 observed variables + 1 hidden confounder

## License

[As specified by course/institution]

---

**Last Updated:** April 2026