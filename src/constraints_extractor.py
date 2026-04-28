"""
Constraints Extractor: Derive connection constraints from data analysis.

This module provides utilities to extract connection constraints from correlation
and mutual information analysis, which can then be used to filter valid DAGs.

A "connection" between two variables means they are not d-separated (i.e., they
have some statistical relationship that cannot be explained by confounding alone).
"""

from typing import List, Tuple, Optional
import pandas as pd
import numpy as np


def extract_from_correlations(
    correlations_df: pd.DataFrame,
    threshold: float = 0.1,
    columns: Tuple[str, str, str] = ("var1", "var2", "pearson_r"),
) -> List[Tuple[str, str]]:
    """
    Extract connection constraints from a correlation DataFrame.

    Identifies variable pairs with correlation magnitude exceeding the threshold.
    These pairs are considered "connected" (not d-separated).

    Args:
        correlations_df: DataFrame with columns for variable names and correlation values.
                        Must contain columns matching the 'columns' parameter.
                        Example output from pairwise_linear_correlations():
                        | var1 | var2 | pearson_r |
                        | A    | B    | 0.85      |
                        | C    | D    | 0.42      |
        threshold: Absolute correlation magnitude threshold. Pairs with |r| >= threshold
                  are considered connected. Default: 0.1
        columns: Tuple of (var1_col, var2_col, corr_col) naming the relevant columns.
                Default: ("var1", "var2", "pearson_r")

    Returns:
        List of (var1, var2) tuples representing connected variable pairs.
        Normalized so var1 < var2 alphabetically for consistency.

    Example:
        >>> import pandas as pd
        >>> corr_df = pd.DataFrame({
        ...     'var1': ['A', 'B'],
        ...     'var2': ['B', 'C'],
        ...     'pearson_r': [0.85, 0.05]
        ... })
        >>> constraints = extract_from_correlations(corr_df, threshold=0.1)
        >>> constraints
        [('A', 'B')]
    """
    var1_col, var2_col, corr_col = columns
    
    connections = []
    for _, row in correlations_df.iterrows():
        if abs(row[corr_col]) >= threshold:
            v1, v2 = row[var1_col], row[var2_col]
            # Normalize order for consistency
            if v1 > v2:
                v1, v2 = v2, v1
            connections.append((v1, v2))
    
    return connections


def extract_from_mutual_information(
    variables: List[str],
    data: pd.DataFrame,
    threshold: float = 0.05,
    nbins: int = 21,
) -> List[Tuple[str, str]]:
    """
    Extract connection constraints from mutual information analysis.

    Computes pairwise mutual information between all variable pairs and identifies
    those exceeding the threshold.

    Args:
        variables: List of variable names to analyze, e.g., ['A', 'B', 'C', 'D']
        data: DataFrame containing the data. Must have columns matching variable names.
        threshold: Mutual information threshold. Pairs with MI >= threshold are
                  considered connected. Default: 0.05
        nbins: Number of bins for histogram-based MI estimation. Default: 21

    Returns:
        List of (var1, var2) tuples representing connected variable pairs.
        Normalized so var1 < var2 alphabetically for consistency.

    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>> data = pd.DataFrame({
        ...     'A': np.random.randn(100),
        ...     'B': np.random.randn(100),
        ...     'C': np.random.randn(100)
        ... })
        >>> constraints = extract_from_mutual_information(['A', 'B', 'C'], data, threshold=0.01)
        >>> len(constraints)  # Number of connected pairs (varies with data)
        2
    """
    connections = []
    eps = np.spacing(1)
    
    for i in range(len(variables)):
        for j in range(i + 1, len(variables)):
            var_i, var_j = variables[i], variables[j]
            x = data[var_i].values
            y = data[var_j].values
            
            # Compute MI using histogram-based estimation
            mi_value = _compute_mutual_information(x, y, nbins, eps)
            
            if mi_value >= threshold:
                connections.append((var_i, var_j))
    
    return connections


def _compute_mutual_information(x: np.ndarray, y: np.ndarray, nbins: int, eps: float) -> float:
    """
    Compute mutual information between two variables using histogram-based estimation.

    Args:
        x: First variable (1D array)
        y: Second variable (1D array)
        nbins: Number of bins for histogram
        eps: Small value for numerical stability

    Returns:
        Mutual information value
    """
    bins = np.linspace(np.min(x), np.max(x), nbins)
    
    x_marginal = np.histogram(x, bins=bins)[0]
    x_marginal = x_marginal / x_marginal.sum()
    
    y_marginal = np.histogram(y, bins=bins)[0]
    y_marginal = y_marginal / y_marginal.sum()
    
    xy_joint = np.histogram2d(x, y, bins=(bins, bins))[0]
    xy_joint = xy_joint / xy_joint.sum()
    
    # MI = sum(p(x,y) * log(p(x,y) / (p(x) * p(y))))
    mi = np.sum(
        xy_joint * np.log(xy_joint / (x_marginal[:, None] * y_marginal[None, :] + eps) + eps)
    )
    
    return mi


def combine_constraints(
    constraints_list: List[List[Tuple[str, str]]]
) -> List[Tuple[str, str]]:
    """
    Combine multiple lists of constraints and remove duplicates.

    Args:
        constraints_list: List of constraint lists (e.g., from different methods)

    Returns:
        Combined list with duplicates removed and normalized ordering.

    Example:
        >>> c1 = [('A', 'B'), ('B', 'C')]
        >>> c2 = [('A', 'B'), ('C', 'D')]
        >>> combine_constraints([c1, c2])
        [('A', 'B'), ('B', 'C'), ('C', 'D')]
    """
    all_constraints = set()
    for constraints in constraints_list:
        for v1, v2 in constraints:
            # Normalize ordering
            if v1 > v2:
                v1, v2 = v2, v1
            all_constraints.add((v1, v2))
    
    return sorted(list(all_constraints))
