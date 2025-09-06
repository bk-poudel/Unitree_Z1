import mujoco
import mujoco.viewer as viewer
import numpy as np
import time
from scipy.spatial.transform import Rotation as R, Slerp
from utils import get_states, get_ee_pose, get_jacobian
from pinocchio import RobotWrapper
import csv

# ----------------------
# Simulation Parameters
# ----------------------
dt = 0.001
model = mujoco.MjModel.from_xml_path("./scene.xml")
data = mujoco.MjData(model)
# Load robot URDF in Pinocchio
robot = RobotWrapper.BuildFromURDF(
    "./z1.urdf", "/home/bibek/Unitree_Z1/Unitree_Z1/z1_description/meshes"
)
# Home position
initial_position = np.array([0, 0, 0, 0, 0, 0, 0])
home_pos = np.array([0, 0.785, -0.261, -0.523, 0, 1.57, -1.5])
data.qpos[:7] = initial_position
model.opt.gravity[2] = -9.81
mujoco.mj_forward(model, data)
# Launch viewer
viewer = viewer.launch_passive(model, data)
# ----------------------
# Joint limits
# ----------------------
joint_limits = []
for j in range(model.njnt):
    if model.jnt_limited[j]:
        joint_limits.append(model.jnt_range[j])
    else:
        joint_limits.append([-np.inf, np.inf])
joint_limits = np.array(joint_limits[:6])  # arm joints only
# ----------------------
# Torque control gains
# ----------------------
kp = np.array([1000, 2000, 1000, 1000, 1000, 1000])
kd = np.array([200, 200, 150, 100, 50, 50])


# ----------------------
# Inverse Kinematics
# ----------------------
def inverse_kinematics(q_init, target_pos, target_ori, robot, max_iters=100, tol=1e-4):
    q_current = q_init.copy()
    q_ref = (joint_limits[:, 0] + joint_limits[:, 1]) / 2.0
    for _ in range(max_iters):
        current_ee_pose = get_ee_pose(q_current, robot)
        pos_error = target_pos - current_ee_pose[:3, 3]
        R_error = target_ori @ current_ee_pose[:3, :3].T
        ori_error = R.from_matrix(R_error).as_rotvec()
        error = np.concatenate([pos_error, ori_error])
        if np.linalg.norm(error) < tol:
            break
        J = get_jacobian(robot, q_current)
        J_pos = current_ee_pose[:3, :3] @ J[:3, :]
        J_ori = current_ee_pose[:3, :3] @ J[3:6, :]
        J_full = np.vstack([J_pos, J_ori])
        J_pinv = np.linalg.pinv(J_full)
        dq_task = J_pinv @ error
        dq_null = (np.eye(len(q_current)) - J_pinv @ J_full) @ (
            0.1 * (q_ref[:6] - q_current)
        )
        dq = dq_task + dq_null
        q_current += dq
        for i in range(len(q_current)):
            q_current[i] = np.clip(q_current[i], joint_limits[i, 0], joint_limits[i, 1])
    return q_current


# ----------------------
# Orientation interpolation
# ----------------------
def interpolate_orientations(R_start, R_end, num_steps=100):
    key_times = [0, 1]
    rotations = R.from_matrix([R_start, R_end])
    slerp = Slerp(key_times, rotations)
    times = np.linspace(0, 1, num_steps)
    interp_rots = slerp(times)
    return [r.as_matrix() for r in interp_rots]


# ----------------------
# Task-space waypoints
# ----------------------
q, dq = get_states(data)
initial_pose = get_ee_pose(q, robot)
initial_position = initial_pose[:3, 3]
initial_orientation = initial_pose[:3, :3]
bottle_pos = np.array([0.42, 0, 0.2])
glass_pos = np.array([0.4, 0.3, 0.2])
waypoints = [
    initial_position,
    initial_position + np.array([0, 0, 0.2]),
    bottle_pos + np.array([0, 0, 0.2]),
    bottle_pos,
    bottle_pos + np.array([0, 0, 0.2]),
    glass_pos + np.array([0, 0, 0.2]),
    glass_pos,
]
rotated_orientation = (
    initial_orientation @ R.from_euler("x", 90, degrees=True).as_matrix()
)
orientations = [
    initial_orientation,
    initial_orientation,
    rotated_orientation,
    rotated_orientation,
    rotated_orientation,
    rotated_orientation,
    rotated_orientation,
]
pour_orientation = initial_orientation @ R.from_euler("x", 0, degrees=True).as_matrix()
waypoints.append(glass_pos)
orientations.append(pour_orientation)
waypoints.append(glass_pos + np.array([0, 0, 0.2]))
orientations.append(rotated_orientation)
# ----------------------
# Interpolate trajectory
# ----------------------
trajectory = []
trajectory_orient = []
for i in range(len(waypoints) - 1):
    p_start, p_end = waypoints[i], waypoints[i + 1]
    R_start, R_end = orientations[i], orientations[i + 1]
    interp_positions = [
        (1 - alpha) * p_start + alpha * p_end for alpha in np.linspace(0, 1, 2000)
    ]
    interp_orientations = interpolate_orientations(R_start, R_end, num_steps=2000)
    trajectory.extend(interp_positions)
    trajectory_orient.extend(interp_orientations)
# ----------------------
# Main loop (torque control)
# ----------------------
gripper_open = -1.5
gripper_closed = -0.0
with open("joint_states.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([f"q{i+1}" for i in range(7)] + [f"dq{i+1}" for i in range(7)])
for step in range(len(trajectory)):
    target_pos = trajectory[step]
    target_ori = trajectory_orient[step]
    q, dq = get_states(data)
    # Save q and dq
    with open("joint_states.csv", "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(list(q) + list(dq))
    # IK
    q_des = inverse_kinematics(q, target_pos, target_ori, robot)
    dq_des = np.zeros(6)
    # Torque control
    tau = kp * (q_des[:6] - q[:6]) + kd * (dq_des - dq[:6])
    data.qfrc_applied[:6] = tau
    # Gripper control
    if step < 2000:
        data.qpos[6] = 0
    elif step == 2000:
        for i in range(200):
            data.qfrc_applied[:6] = tau
            # Linear interpolation for gripper position
            data.qpos[6] = -1.5 + (i / 199) * (-0.8 + 1.5)
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(dt)
    elif step > 2000 and step < 6000:
        data.qpos[6] = -1.5
    else:
        data.qpos[6] = -0.8
    if step == 6000:  # at bottle grasping point
        # Slowly close gripper from -1.5 to -0.3 over 200 timesteps
        for i in range(200):
            data.qfrc_applied[:6] = tau
            # Linear interpolation for gripper position
            data.qpos[6] = -1.5 + (i / 199) * (-0.8 + 1.5)
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(dt)
    mujoco.mj_step(model, data)
    viewer.sync()
    time.sleep(dt)
viewer.close()
