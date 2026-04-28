from pathlib import Path
import numpy as np
from scipy.stats import gaussian_kde
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def get_csv_file_list():
    data_dir = Path(__file__).resolve().parent / "data"
    print(data_dir)
    return [str(path) for path in data_dir.glob("*.csv")]

def get_csv_file(csv_file_name):
    data_dir = Path(__file__).resolve().parent / "data"
    print(data_dir)
    return Path.joinpath(data_dir, csv_file_name)

def load_csv_data(csv_file_path):
    return pd.read_csv(csv_file_path)
    
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

def plot_mutual_information():
    pass

if __name__ == "__main__":
    #first get a list of all csv files in the data directory

    #then loop thorugh to get the mutual information for each pair of variables in each csv file and print the results
    
    #do that for all csv files in the data directory
    
    #Compare each csv result with eachother to see if there are any patterns or differences in the mutual information values across different datasets
    
    path = "./data/-2a.csv"
    df = load_csv_data(path)

    b = df["B"].to_numpy()
    b_x = np.linspace(min(b), max(b), len(b))
    print(b, b.shape)
    c = df["C"].to_numpy()
    c_x = np.linspace(min(c), max(c), len(c))
    print(c, c.shape)
    omega=np.random.rand(100)*6*np.pi
    #print(kde_MI(a, b))
    
    
    print('Correlation between b and c: %.2f'%np.corrcoef(b,c)[0,1]);
    
    sns.jointplot(x=b,y=c,kind='kde');
    plt.show()