import csv
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

csv_file = "joint_states.csv"
joint_states = []
joint_velocities = []
torques_all = []
dt = 0.01
kp = [20, 30, 30, 20, 15, 10]
kd = [2000, 2000, 2000, 2000, 2000, 2000]
# --- Read CSV ---
with open(csv_file, newline="") as f:
    reader = csv.reader(f)
    for row in reader:
        data = [float(x) for x in row]
        joint_states.append(data[:6])
        joint_velocities.append(data[6:])  # optional, from CSV
joint_states = np.array(joint_states)
joint_velocities = np.array(joint_velocities)
# --- Compute numerical derivatives ---
derived_velocities = np.gradient(joint_states, dt, axis=0)
# --- Smooth velocities ---
qd_smoothed = np.zeros_like(derived_velocities)
for i in range(6):
    qd_smoothed[:, i] = savgol_filter(
        derived_velocities[:, i], window_length=7, polyorder=2
    )
# --- Compute torques using PD formula ---
for idx in range(len(joint_states)):
    q = joint_states[idx]
    dq = qd_smoothed[idx]  # use smoothed velocity
    if idx < len(joint_states) - 1:
        qd = joint_states[idx + 1]  # next desired position
        dqd = qd_smoothed[idx + 1]
    else:
        qd = joint_states[idx]
        dqd = qd_smoothed[idx]
    torque = [
        kp[i] * 25.6 * (qd[i] - q[i]) + kd[i] * 0.0128 * (dqd[i] - dq[i])
        for i in range(6)
    ]
    torques_all.append(torque)
torques_all = np.array(torques_all)
# --- Plot smoothed velocities ---
plt.figure(figsize=(10, 5))
for i in range(6):
    plt.plot(qd_smoothed[:, i], label=f"Smoothed dq {i+1}")
plt.title("Smoothed Joint Velocities")
plt.xlabel("Sample")
plt.ylabel("Velocity")
plt.legend()
plt.grid(True)
plt.savefig("smoothed_joint_velocities.png")
plt.show()
# --- Plot torques ---
plt.figure(figsize=(10, 5))
for i in range(6):
    plt.plot(torques_all[:, i], label=f"Torque {i+1}")
plt.title("Joint Torques")
plt.xlabel("Sample")
plt.ylabel("Torque")
plt.legend()
plt.grid(True)
plt.ylim(-3, 3)
plt.savefig("joint_torques.png")
plt.show()
