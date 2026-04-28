"""
Main workflow for causal graph inference.

This module orchestrates the full pipeline:
1. Load observational data from CSV
2. Perform correlation analysis (linear)
3. Perform mutual information analysis (nonlinear)
4. Extract connection constraints from both analyses
5. Find all valid DAG structures consistent with constraints
6. Display and save results

Usage:
    python main.py

Configuration:
    Edit config.py to change thresholds, data paths, and node parameters.
"""

import os
import json
import pandas as pd
import numpy as np

import config
from graph_finder import find_valid_graphs
from constraints_extractor import (
    extract_from_correlations,
    extract_from_mutual_information,
    combine_constraints,
)


def load_data(filepath: str) -> pd.DataFrame:
    """Load CSV data from file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    df = pd.read_csv(filepath)
    if config.VERBOSE:
        print(f"✓ Loaded data from {filepath}")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
    return df


def analyze_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Compute pairwise Pearson correlations for numeric columns."""
    numeric_df = df.select_dtypes(include=["number"])
    cols = list(numeric_df.columns)
    
    rows = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            col_a = cols[i]
            col_b = cols[j]
            corr = numeric_df[col_a].corr(numeric_df[col_b], method="pearson")
            rows.append({"var1": col_a, "var2": col_b, "pearson_r": corr})
    
    result = pd.DataFrame(rows).sort_values(
        by="pearson_r", key=lambda s: s.abs(), ascending=False
    )
    
    if config.VERBOSE:
        print(f"\n✓ Correlation Analysis ({len(result)} pairs):")
        print(f"  Threshold: {config.CORRELATION_THRESHOLD}")
        print("  Top 5 correlations by magnitude:")
        for _, row in result.head(5).iterrows():
            print(f"    {row['var1']} -- {row['var2']}: {row['pearson_r']:.4f}")
    
    return result


def get_variables_from_data(df: pd.DataFrame) -> list:
    """Extract numeric column names from dataframe (these are the observed variables)."""
    return list(df.select_dtypes(include=["number"]).columns)


def extract_constraints(
    df: pd.DataFrame, correlations_df: pd.DataFrame
) -> list:
    """
    Extract connection constraints from data analysis.
    
    Combines constraints from both correlation and mutual information analysis.
    """
    # Extract from correlations
    corr_constraints = extract_from_correlations(
        correlations_df,
        threshold=config.CORRELATION_THRESHOLD,
        columns=("var1", "var2", "pearson_r"),
    )
    
    # Extract from mutual information
    observed_vars = get_variables_from_data(df)
    mi_constraints = extract_from_mutual_information(
        observed_vars,
        df,
        threshold=config.MI_THRESHOLD,
        nbins=config.MI_NBINS,
    )
    
    # Combine and deduplicate
    all_constraints = combine_constraints([corr_constraints, mi_constraints])
    
    if config.VERBOSE:
        print(f"\n✓ Constraint Extraction:")
        print(f"  Correlation constraints: {len(corr_constraints)}")
        print(f"  Mutual information constraints: {len(mi_constraints)}")
        print(f"  Combined (deduplicated): {len(all_constraints)}")
        if all_constraints:
            print(f"  Connection pairs: {all_constraints}")
    
    return all_constraints


def find_graphs(observed_vars: list, constraints: list) -> list:
    """
    Find all valid DAG structures consistent with constraints.
    """
    if config.VERBOSE:
        print(f"\n✓ Finding Valid DAG Structures:")
        print(f"  Observed variables: {observed_vars}")
        print(f"  Total nodes (including _H): {len(observed_vars) + 1}")
        print(f"  Constraints to satisfy: {len(constraints)}")
        if config.HIDDEN_CONFOUND_PAIRS:
            print(f"  Specific confound pair: {config.HIDDEN_CONFOUND_PAIRS}")
        else:
            print(f"  Searching all possible confound pairs...")
        print()
    
    graphs = find_valid_graphs(
        observed_vars,
        constraints,
        h_confounds=config.HIDDEN_CONFOUND_PAIRS,
    )
    
    if config.VERBOSE:
        print(f"\n✓ Found {len(graphs)} valid DAG configurations")
    
    return graphs


def save_results(graphs: list, constraints: list, correlations: pd.DataFrame) -> None:
    """Save results to output files."""
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # Save graphs as JSON
    graphs_json = [
        {"edges": [(u, v) for u, v in graph]} for graph in graphs
    ]
    graphs_file = os.path.join(config.OUTPUT_DIR, config.GRAPHS_OUTPUT_FILE)
    with open(graphs_file, "w") as f:
        json.dump(graphs_json, f, indent=2)
    if config.VERBOSE:
        print(f"\n✓ Saved {len(graphs)} graphs to {graphs_file}")
    
    # Save constraints
    constraints_file = os.path.join(config.OUTPUT_DIR, config.CONSTRAINTS_OUTPUT_FILE)
    with open(constraints_file, "w") as f:
        json.dump({"constraints": constraints}, f, indent=2)
    if config.VERBOSE:
        print(f"✓ Saved constraints to {constraints_file}")
    
    # Save correlation analysis
    corr_file = os.path.join(config.OUTPUT_DIR, config.ANALYSIS_OUTPUT_FILE)
    correlations.to_csv(corr_file, index=False)
    if config.VERBOSE:
        print(f"✓ Saved correlation analysis to {corr_file}")


def main():
    """Run the full causal inference pipeline."""
    print("=" * 70)
    print("AL3-MiniProject: Causal Graph Inference")
    print("=" * 70)
    
    try:
        # Step 1: Load data
        data = load_data(config.TEST_DATA_PATH)
        
        # Step 2: Analyze correlations
        correlations = analyze_correlations(data)
        
        # Step 3: Extract constraints
        observed_variables = get_variables_from_data(data)
        constraints = extract_constraints(data, correlations)
        
        # Step 4: Find valid graphs
        graphs = find_graphs(observed_variables, constraints)
        
        # Step 5: Save results
        save_results(graphs, constraints, correlations)
        
        # Summary
        print("\n" + "=" * 70)
        print("Pipeline completed successfully!")
        print(f"  Total valid DAGs found: {len(graphs)}")
        print(f"  Connection constraints: {len(constraints)}")
        print(f"  Output directory: {config.OUTPUT_DIR}/")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
