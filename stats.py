import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_variables(df, variables):
    """
    Plots the distribution of specified variables in the DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the data.
    variables (list): A list of column names to plot.

    Returns:
    None: Displays the plots.
    """
    for var in variables:
        plt.figure(figsize=(10, 6))
        sns.histplot(df[var], kde=True)
        plt.title(f'Distribution of {var}')
        plt.xlabel(var)
        plt.ylabel('Frequency')
        plt.show()