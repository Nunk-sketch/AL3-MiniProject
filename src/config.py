"""
Project Configuration: Central location for all project parameters.
"""

# ============ Observed Variables ============
# For testing: use 4 variables (5 nodes total with _H)
TEST_OBSERVED_NODES = ["A", "B", "C", "D"]
TEST_TOTAL_NODES = 5  # 4 observed + 1 hidden

# For actual analysis: use 6 variables (7 nodes total with _H)
ACTUAL_OBSERVED_NODES = ["A", "B", "C", "D", "E", "F"]
ACTUAL_TOTAL_NODES = 7  # 6 observed + 1 hidden

# Active configuration (set to TEST or ACTUAL as needed)
OBSERVED_NODES = TEST_OBSERVED_NODES
TOTAL_NODES = TEST_TOTAL_NODES

# ============ Data Files ============
DATA_DIR = "data"
TEST_DATA_FILE = "data_1504.csv"
TEST_DATA_PATH = f"{DATA_DIR}/{TEST_DATA_FILE}"

# ============ Analysis Parameters ============
# Correlation threshold for identifying connections
CORRELATION_THRESHOLD = 0.1

# Mutual information threshold for identifying connections
MI_THRESHOLD = 0.05

# Number of bins for histogram-based MI estimation
MI_NBINS = 21

# ============ Cost Parameters ============
# These are used for the active learning / intervention planning phase
COST_PER_SAMPLE = 1  # Cost of observing a new sample
COST_PER_INCORRECT_GUESS = 70  # Cost of incorrectly inferring a graph structure
COST_PER_EXPERIMENT = 20  # Cost of running one experimental intervention

# Intervention bounds: interventions are clamped to [-2, 2]
INTERVENTION_MIN = -2
INTERVENTION_MAX = 2

# ============ Output Parameters ============
OUTPUT_DIR = "outputs"
GRAPHS_OUTPUT_FILE = "valid_graphs.json"
CONSTRAINTS_OUTPUT_FILE = "extracted_constraints.json"
ANALYSIS_OUTPUT_FILE = "correlation_analysis.csv"

# ============ Hidden Confounder Configuration ============
# Set to None to search all possible pairs, or specify a tuple like ("A", "B")
HIDDEN_CONFOUND_PAIRS = None  # Search all pairs

# ============ Logging ============
VERBOSE = True  # Print progress information
