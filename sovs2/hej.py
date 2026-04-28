from pathlib import Path
import numpy as np
from scipy.stats import gaussian_kde
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import gaussian_kde

def get_csv_file_list(): #returns a list of all csv files in the data directory
    data_dir = "./data"
    print(data_dir)
    return [str(path) for path in Path(data_dir).glob("*.csv")]

def get_csv_file(csv_file_name): #takes a csv file name and returns the full path to the file if it exists in the data directory, otherwise returns None
    tempList = get_csv_file_list()
    for file in tempList:
        if csv_file_name in file:
            return file
    return None

def load_csv_data(csv_file_name): #takes a csv file name and returns a pandas dataframe
    csv_file_path = get_csv_file(csv_file_name)
    if csv_file_path:
        return pd.read_csv(csv_file_path)
    else:
        return None

def get_columns(dataframe):
    return dataframe.columns.tolist()
    
def kde_MI(x, y, grid_points=30):
    # Ensure inputs are numpy arrays
    x = np.asarray(x)
    y = np.asarray(y)
    # Create the KDE for the joint distribution
    values = np.vstack([x, y])
    kernel = gaussian_kde(values)
    # Define the grid over which to evaluate the densities
    x_grid = np.linspace(x.min(), x.max(), grid_points)
    y_grid = np.linspace(y.min(), y.max(), grid_points)
    X, Y = np.meshgrid(x_grid, y_grid)
    positions = np.vstack([X.ravel(), Y.ravel()])
    # Evaluate joint density P(x,y)
    p_xy = kernel(positions).reshape(grid_points, grid_points)
    # Normalize to ensure the sum equals 1 (forming a probability mass
    p_xy /= np.sum(p_xy)
    # Calculate marginal densities P(x) and P(y) by summing the joint
    p_x = np.sum(p_xy, axis=0)
    p_y = np.sum(p_xy, axis=1)
    # Calculate P(x)P(y) using an outer product
    p_x_p_y = p_x[np.newaxis, :] * p_y[:, np.newaxis]
    # Calculate Mutual Information
    mask = p_xy > 1e-12 # Use a small threshold for numerical stabili
    mi = np.sum(p_xy[mask] * np.log(p_xy[mask] / p_x_p_y[mask]))
    return mi

def get_mutual_information():
    pass

def plot_mutual_information(dataframe):
    df = dataframe.iloc[:, 1:]  # Ignore the first column
    cols = get_columns(df)
    n = len(cols)

    fig, axes = plt.subplots(n, n, figsize=(3.5 * n, 3.5 * n))
    axes = np.atleast_2d(axes)

    for i in range(n):
        for j in range(n):
            ax = axes[i, j]

            x = pd.to_numeric(df[cols[j]], errors="coerce").dropna().to_numpy()
            y = pd.to_numeric(df[cols[i]], errors="coerce").dropna().to_numpy()

            if len(x) == 0 or len(y) == 0 or np.std(x) == 0 or np.std(y) == 0:
                ax.axis("off")
                continue

            divider = make_axes_locatable(ax)
            ax_top = divider.append_axes("top", size="22%", pad=0.05, sharex=ax)
            ax_right = divider.append_axes("right", size="22%", pad=0.05, sharey=ax)

            if i == j:
                # Linear plot for the same column
                ax.plot(x, x, color="steelblue", linewidth=1.2)
            else:
                # Compute mutual information and create KDE contour plot
                mi = kde_MI(x, y)
                
                # Create KDE contour plot
                xx, yy = np.mgrid[x.min():x.max():100j, y.min():y.max():100j]
                positions = np.vstack([xx.ravel(), yy.ravel()])
                kernel = gaussian_kde(np.vstack([x, y]))
                f = np.reshape(kernel(positions).T, xx.shape)
                ax.contour(xx, yy, f, colors="steelblue", linewidths=1.2, alpha=0.8)
                
                ax.text(0.05, 0.95, f'MI: {mi:.3f}', transform=ax.transAxes, 
                        fontsize=8, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            # Plot marginal distributions as density lines
            x_kde = gaussian_kde(x)
            y_kde = gaussian_kde(y)
            
            x_range = np.linspace(x.min(), x.max(), 100)
            y_range = np.linspace(y.min(), y.max(), 100)
            
            ax_top.plot(x_range, x_kde(x_range), color="steelblue", linewidth=1.5)
            ax_right.plot(y_kde(y_range), y_range, color="steelblue", linewidth=1.5)

            ax_top.axis("off")
            ax_right.axis("off")

            ax.set_xlabel(cols[j], fontsize=8)
            ax.set_ylabel(cols[i], fontsize=8)
            ax.tick_params(labelsize=6)

    plt.tight_layout()
    plt.show()
    

if __name__ == "__main__":
    #first get a list of all csv files in the data directory

    #then loop thorugh to get the mutual information for each pair of variables in each csv file and print the results
    
    #do that for all csv files in the data directory
    
    #Compare each csv result with eachother to see if there are any patterns or differences in the mutual information values across different datasets
    plot_mutual_information(load_csv_data("data_1504.csv"))
