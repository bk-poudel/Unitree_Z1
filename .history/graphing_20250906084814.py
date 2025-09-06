import csv
import matplotlib.pyplot as plt

csv_file = "joint states.csv"
joint_states = []
joint_velocities = []
with open(csv_file, newline="") as f:
    reader = csv.reader(f)
    for row in reader:
        # Convert all values to float
        data = [float(x) for x in row]
        joint_states.append(data[:6])
        joint_velocities.append(data[6:])
# Plot joint states
plt.figure(figsize=(10, 5))
for i in range(6):
    plt.plot([js[i] for js in joint_states], label=f"Joint State {i+1}")
plt.title("Joint States")
plt.xlabel("Sample")
plt.ylabel("State Value")
plt.legend()
plt.show()
# Plot joint velocities
plt.figure(figsize=(10, 5))
for i in range(len(joint_velocities[0])):
    plt.plot([jv[i] for jv in joint_velocities], label=f"Joint Velocity {i+1}")
plt.title("Joint Velocities")
plt.xlabel("Sample")
plt.ylabel("Velocity Value")
plt.legend()
plt.show()
