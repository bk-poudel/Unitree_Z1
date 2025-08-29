import time
import numpy as np
from Z1Sim import Z1Sim

xml_path = "/home/bibek/Unitree_Z1/Unitree_Z1/scene.xml"
sim = Z1Sim(xml_path, dt=0.001)
Kp_cart = np.array([800, 800, 800])
Kd_cart = np.array([120, 120, 120])
Kp_joint = np.array([50] * 6)
Kd_joint = np.array([5] * 6)
q_current = sim.data.qpos[:6].copy()
x_start = sim.forward_kinematics(q_current)[:3, 3]
# Rectangle corners (relative to x_start)
rect = [
    x_start,
    x_start + np.array([0.2, 0, 0]),
    x_start + np.array([0.2, 0.1, 0]),
    x_start + np.array([0, 0.1, 0]),
    x_start,
]
T_total = 2.0  # seconds per segment
dt = sim.dt
steps = int(T_total / dt)
for seg in range(len(rect) - 1):
    x0, x1 = rect[seg], rect[seg + 1]
    for i in range(steps):
        q, dq = sim.get_state()
        T_current = sim.forward_kinematics(q[:6])
        R_ee = T_current[:3, :3]
        x_current = T_current[:3, 3]
        x_des = x0 + (x1 - x0) * (i / steps)
        J = sim.get_jacobian(q[:6])
        J_spatial = np.block([[R_ee @ J[3:, :]]])
        pos_error = x_des - x_current
        vel_error = -(J_spatial @ dq[:6])
        Kp = np.diag(Kp_cart)
        Kd = np.diag(Kd_cart)
        tau = J_spatial.T @ (Kp @ pos_error + Kd @ vel_error)
        sim.send_joint_torque(tau)
        time.sleep(dt)
