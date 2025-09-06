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
dt = 0.001  # simulation timestep
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
q, dq = get_states(data)
# Launch viewer
viewer = viewer.launch_passive(model, data)
# ----------------------
# Joint limits from MuJoCo
# ----------------------
joint_limits = []
for j in range(model.njnt):
    if model.jnt_limited[j]:
        joint_limits.append(model.jnt_range[j])
    else:
        joint_limits.append([-np.inf, np.inf])
joint_limits = np.array(joint_limits)
joint_limits = joint_limits[:6]  # only arm joints


# ----------------------
# Inverse Kinematics with joint limit handling
# ----------------------
def inverse_kinematics(
    q_init, target_pos, target_ori, robot, max_iters=100, tol=1e-4, alpha=0.5
):
    q_current = q_init.copy()
    q_ref = (joint_limits[:, 0] + joint_limits[:, 1]) / 2.0  # mid-range reference
    for _ in range(max_iters):
        current_ee_pose = get_ee_pose(q_current, robot)
        pos_error = target_pos - current_ee_pose[:3, 3]
        R_error = target_ori @ current_ee_pose[:3, :3].T
        ori_error = R.from_matrix(R_error).as_rotvec()
        error = np.concatenate([pos_error, ori_error])
        if np.linalg.norm(error) < tol:
            break
        # Jacobian
        J = get_jacobian(robot, q_current)
        J_pos = current_ee_pose[:3, :3] @ J[:3, :]
        J_ori = current_ee_pose[:3, :3] @ J[3:6, :]
        J_full = np.vstack([J_pos, J_ori])
        # Pseudoinverse
        J_pinv = np.linalg.pinv(J_full)
        # Nullspace projection to stay near reference posture
        dq_task = J_pinv @ error
        dq_null = (np.eye(len(q_current)) - J_pinv @ J_full) @ (
            0.1 * (q_ref[:6] - q_current)
        )
        dq = dq_task + dq_null
        q_current += dq
        # Clamp to joint limits
        for i in range(len(q_current)):
            q_current[i] = np.clip(q_current[i], joint_limits[i, 0], joint_limits[i, 1])
    return q_current


# ----------------------
# Orientation interpolation using Slerp
# ----------------------
def interpolate_orientations(R_start, R_end, num_steps=100):
    key_times = [0, 1]
    rotations = R.from_matrix([R_start, R_end])
    slerp = Slerp(key_times, rotations)
    times = np.linspace(0, 1, num_steps)
    interp_rots = slerp(times)
    return [r.as_matrix() for r in interp_rots]


# ----------------------
# Cartesian interpolation from initial to home position
# ----------------------
# Get initial end-effector pose
q, dq = get_states(data)
initial_ee_pose = get_ee_pose(q[:6], robot)  # Use only first 6 joints for arm
initial_ee_pos = initial_ee_pose[:3, 3]
initial_ee_ori = initial_ee_pose[:3, :3]
# Set robot to home position temporarily to get home end-effector pose
data.qpos[:7] = home_pos
mujoco.mj_forward(model, data)
q_home, _ = get_states(data)
home_ee_pose = get_ee_pose(q_home[:6], robot)
home_ee_pos = home_ee_pose[:3, 3]
home_ee_ori = home_ee_pose[:3, :3]
# Reset to initial position
data.qpos[:7] = initial_position
mujoco.mj_forward(model, data)
# Create Cartesian trajectory from initial to home
num_steps = int(2 / dt)
interp_positions = []
interp_orientations = []
for i in range(num_steps):
    alpha = i / (num_steps - 1)
    # Linear interpolation for position
    interp_pos = (1 - alpha) * initial_ee_pos + alpha * home_ee_pos
    interp_positions.append(interp_pos)
# Spherical linear interpolation for orientation
interp_orientations = interpolate_orientations(initial_ee_ori, home_ee_ori, num_steps)
# Initialize CSV file
with open("joint_states.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
# Execute Cartesian trajectory using inverse kinematics
for i in range(num_steps):
    q, dq = get_states(data)
    # Save current joint states to CSV
    with open("joint_states.csv", "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(list(q) + list(dq))
    # Get target Cartesian pose
    target_pos = interp_positions[i]
    target_ori = interp_orientations[i]
    # Compute desired joint angles using inverse kinematics
    q_des = inverse_kinematics(q[:6], target_pos, target_ori, robot)
    # Set arm joint positions
    data.qpos[:6] = q_des
    # Keep gripper position as in home_pos
    data.qpos[6] = home_pos[6]
    mujoco.mj_step(model, data)
    viewer.sync()
    time.sleep(dt)
# Pause at home position
time.sleep(2)
# ----------------------
# Task-space waypoints (rest of your original code)
# ----------------------
q, dq = get_states(data)
initial_pose = get_ee_pose(q, robot)
initial_position = initial_pose[:3, 3]
initial_orientation = initial_pose[:3, :3]
# Bottle and glass positions (x, y, z)
bottle_pos = np.array([0.42, 0, 0.2])
glass_pos = np.array([0.4, 0.3, 0.2])
waypoints = [
    initial_position,
    bottle_pos + np.array([0, 0, 0.2]),  # hover above bottle
    bottle_pos,  # descend to bottle (GRASP HERE)
    bottle_pos + np.array([0, 0, 0.2]),  # lift bottle
    glass_pos + np.array([0, 0, 0.2]),  # move above glass
    glass_pos,  # align with glass
]
orientations = [
    initial_orientation,
    initial_orientation,
    initial_orientation,
    initial_orientation,
    initial_orientation,
    initial_orientation,
]
# Pour orientation: rotate wrist around x-axis ~ 90 deg
pour_orientation = (
    initial_orientation @ R.from_euler("x", -90, degrees=True).as_matrix()
)
waypoints.append(glass_pos)
orientations.append(pour_orientation)
# Return upright
waypoints.append(glass_pos + np.array([0, 0, 0.2]))
orientations.append(initial_orientation)
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
# Main loop
# ----------------------
cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "bottle_grasp")
for step in range(len(trajectory)):
    target_pos = trajectory[step]
    target_ori = trajectory_orient[step]
    q, dq = get_states(data)
    # Save q and dq to CSV
    with open("joint_states.csv", "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Write joint angles (q) and velocities (dq) as separate columns
        writer.writerow(list(q) + list(dq))
    q_des = inverse_kinematics(q, target_pos, target_ori, robot)
    if step < 4000:
        data.qpos[6] = -1.5
    else:
        data.qpos[6] = -0.3
    # Control arm joints
    data.qpos[:6] = q_des[:6]
    # ---- Close gripper when reaching bottle ----
    if step == 4000:  # at bottle grasping point
        # Slowly close gripper from -1.5 to -0.3 over 200 timesteps
        for i in range(200):
            data.qpos[:6] = q_des[:6]  # keep arm steady
            # Linear interpolation for gripper position
            data.qpos[6] = -1.5 + (i / 199) * (-0.3 + 1.5)
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(dt)
    mujoco.mj_step(model, data)
    viewer.sync()
    time.sleep(dt)
viewer.close()
