import time
import numpy as np
from Z1Sim import Z1Sim

xml_path = "/home/bibek/Unitree_Z1/Unitree_Z1/scene.xml"
sim = Z1Sim(xml_path, dt=0.001)
Kp_cart = np.array([800, 800, 800])
Kd_cart = np.array([120, 120, 120])
Kp_ori = np.array([200, 200, 200])  # Orientation gains
Kd_ori = np.array([30, 30, 30])
Kp_joint = np.array([50] * 6)
Kd_joint = np.array([5] * 6)
q_current = sim.data.qpos[:6].copy()
T_start = sim.forward_kinematics(q_current)
x_start = T_start[:3, 3]
R_start = T_start[:3, :3]  # Store initial orientation
rect = [
    x_start,
    x_start + np.array([0.2, 0, 0]),
    x_start + np.array([0.2, 0.1, 0]),
    x_start + np.array([0, 0.1, 0]),
    x_start,
]
T_total = 2.0
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
        R_des = R_start  # Keep orientation constant
        # Position error
        pos_error = x_des - x_current
        # Orientation error (rotation vector)
        R_err = R_des @ R_ee.T
        ori_error = 0.5 * (
            np.cross(R_ee[:, 0], R_des[:, 0])
            + np.cross(R_ee[:, 1], R_des[:, 1])
            + np.cross(R_ee[:, 2], R_des[:, 2])
        )
        J = sim.get_jacobian(q[:6])
        J_pos = J[3:, :]  # Linear part
        J_ori = J[:3, :]  # Angular part
        # Spatial Jacobian
        J_spatial = np.vstack((J_pos, J_ori))
        # Velocity error
        vel_error = -J_spatial @ dq[:6]
        # Gains
        Kp = np.diag(np.concatenate([Kp_cart, Kp_ori]))
        Kd = np.diag(np.concatenate([Kd_cart, Kd_ori]))
        # Combined error
        error = np.concatenate([pos_error, ori_error])
        tau = J_spatial.T @ (Kp @ error + Kd @ vel_error)
        sim.send_joint_torque(tau)
        time.sleep(dt)
