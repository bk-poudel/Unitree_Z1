# test_z1sim_constant_torque.py
import time
import numpy as np
from Z1Sim import Z1Sim  # Make sure Z1Sim.py is in the same folder or in PYTHONPATH

# Path to your MuJoCo XML
xml_path = "/home/bibek/Unitree_Z1/Unitree_Z1/scene.xml"
# Create simulation object
sim = Z1Sim(xml_path, dt=0.001)
# Define constant torque (0.1 Nm) for all 6 joints
# PD gains in Cartesian space
Kp_cart = np.array([800, 800, 800])  # only X direction
Kd_cart = np.array([120, 120, 120])
# PD gains in joint space
Kp_joint = np.array([50] * 6)
Kd_joint = np.array([5] * 6)
# Starting position
q_current = sim.data.qpos[:6].copy()
x_start = sim.forward_kinematics(q_current)[:3, 3]  # XYZ of end-effector
# Linear trajectory in X
x_des_end = x_start + np.array([0.2, 0, 0])  # move 0.2m in X
T_total = 2.0  # seconds
dt = sim.dt
steps = int(T_total / dt)
for i in range(steps):
    q, dq = sim.get_state()
    T_current = sim.forward_kinematics(q[:6])
    R_ee = T_current[:3, :3]
    x_current = T_current[:3, 3]
    # Linear trajectory
    x_des = x_start + (x_des_end - x_start) * (i / steps)
    # Cartesian PD
    J = sim.get_jacobian(q[:6])  # 3x6 position Jacobian
    J_spatial = np.block(
        [
            [R_ee @ J[3:, :]],  # Linear part
        ]
    )
    pos_error = x_des - x_current  # 3x1
    vel_error = -(J_spatial @ dq[:6])  # 3x1
    Kp = np.diag(Kp_cart)
    Kd = np.diag(Kd_cart)
    tau = J_spatial.T @ (Kp @ pos_error + Kd @ vel_error)  # 6x1 torque
    sim.send_joint_torque(tau)
    time.sleep(dt)  # real-time sync
