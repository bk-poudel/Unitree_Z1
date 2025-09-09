import csv
import matplotlib.pyplot as plt

csv_file = "joint_states.csv"
joint_states = []
joint_velocities = []
torques_all = []
dt = 0.01
kp = [20, 30, 30, 20, 15, 10]
kd = [2000, 2000, 2000, 2000, 2000, 2000]
# Read CSV and compute velocities
with open(csv_file, newline="") as f:
    reader = csv.reader(f)
    for row in reader:
        data = [float(x) for x in row]
        joint_states.append(data[:6])
        joint_velocities.append(data[6:])
# Compute derived velocities (numerical differentiation)
derived_velocities_all = []
for idx, q in enumerate(joint_states):
    if idx == 0:
        derived_velocities_all.append([0.0] * 6)
    else:
        derived_velocities_all.append(
            [(q[i] - joint_states[idx - 1][i]) / dt for i in range(6)]
        )
# Compute torques using PD formula
for idx in range(len(joint_states)):
    q = joint_states[idx]
    dq = derived_velocities_all[idx]  # use derived velocities
    if idx < len(joint_states) - 1:
        qd = joint_states[idx + 1]  # next desired position
        dqd = derived_velocities_all[idx + 1]
    else:
        qd = joint_states[idx]  # last sample: use current
        dqd = derived_velocities_all[idx]
    torque = [
        kp[i] * 25.6 * (qd[i] - q[i]) + kd[i] * 0.0128 * (dqd[i] - dq[i])
        for i in range(6)
    ]
    torques_all.append(torque)
# Plot joint states
plt.figure(figsize=(10, 5))
for i in range(6):
    plt.plot([js[i] for js in joint_states], label=f"Joint State {i+1}")
plt.title("Joint States")
plt.xlabel("Sample")
plt.ylabel("State Value")
plt.legend()
plt.savefig("joint_states.png")
plt.show()
# Plot joint velocities (from CSV)
plt.figure(figsize=(10, 5))
for i in range(6):
    plt.plot([jv[i] for jv in joint_velocities], label=f"Joint Velocity {i+1}")
plt.title("Joint Velocities (from CSV)")
plt.xlabel("Sample")
plt.ylabel("Velocity Value")
plt.legend()
plt.savefig("joint_velocities.png")
plt.show()
# Plot derived velocities
plt.figure(figsize=(10, 5))
for i in range(6):
    plt.plot([dv[i] for dv in derived_velocities_all], label=f"Derived Velocity {i+1}")
plt.title("Derived Joint Velocities (Numerical Differentiation)")
plt.xlabel("Sample")
plt.ylabel("Derived Velocity Value")
plt.legend()
plt.savefig("derived_velocities.png")
plt.show()
# Plot torques
plt.figure(figsize=(10, 5))
for i in range(6):
    plt.plot([t[i] for t in torques_all], label=f"Torque {i+1}")
plt.title("Joint Torques")
plt.xlabel("Sample")
plt.ylabel("Torque Value")
plt.legend()
plt.ylim(-3, 3)
plt.yticks([x * 0.5 for x in range(-7, 8)])  # ticks from -3 to 3, step 0.5
plt.savefig("joint_torques.png")
plt.show()
