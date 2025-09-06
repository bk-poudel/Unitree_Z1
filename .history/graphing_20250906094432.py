import csv
import matplotlib.pyplot as plt

csv_file = "joint_states.csv"
joint_states = []
joint_velocities = []
with open(csv_file, newline="") as f:
    reader = csv.reader(f)
    for row in reader:
        # Convert all values to float
        data = [float(x) for x in row]
        joint_states.append(data[:6])
        # Calculate joint velocities using numerical differentiation
        dt = 0.01
        if len(joint_states) > 1:
            prev_state = joint_states[-2]
            curr_state = joint_states[-1]
            derived_velocity = [
                (curr - prev) / dt for curr, prev in zip(curr_state, prev_state)
            ]
        else:
            derived_velocity = [0.0] * 6  # First sample, velocity is zero
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
# Plot derived velocities
plt.figure(figsize=(10, 5))
for i in range(6):
    # Recompute derived velocities for plotting
    derived_velocities = []
    for idx in range(len(joint_states)):
        if idx == 0:
            derived_velocities.append(0.0)
        else:
            derived_velocities.append(
                (joint_states[idx][i] - joint_states[idx - 1][i]) / 0.01
            )
    plt.plot(derived_velocities, label=f"Derived Velocity {i+1}")
plt.title("Derived Joint Velocities (Numerical Differentiation)")
plt.xlabel("Sample")
plt.ylabel("Derived Velocity Value")
plt.legend()
plt.show()
