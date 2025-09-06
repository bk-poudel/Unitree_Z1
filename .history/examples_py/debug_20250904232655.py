import numpy as np

print("Press ctrl+\ to quit process.")
# --- 1. Load Trajectory Data from CSV ---
trajectory_file = "joint_states.csv"
print(f"Loading trajectory from: {trajectory_file}")
try:
    # Load the entire csv file into a numpy array
    trajectory = np.loadtxt(trajectory_file, delimiter=",")
    print(f"Successfully loaded {len(trajectory)} points from trajectory.")
except FileNotFoundError:
    print(f"Error: The file '{trajectory_file}' was not found.")
    print("Please make sure the CSV file is in the same directory as the script.")
    sys.exit(1)
except Exception as e:
    print(f"An error occurred while reading the CSV file: {e}")
    sys.exit(1)
for i in range(len(trajectory)):
    print(trajectory[i])
