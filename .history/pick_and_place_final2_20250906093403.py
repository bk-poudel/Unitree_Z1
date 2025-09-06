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
# Build complete trajectory including home pose movement in Cartesian space
# ----------------------
# Initialize CSV file
with open("joint_states.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
# Get current end-effector pose (at initial position)
data.qpos[:7] = initial_position
mujoco.mj_forward(model, data)
q_initial, _ = get_states(data)
initial_ee_pose = get_ee_pose(q_initial, robot)
initial_ee_position = initial_ee_pose[:3, 3]
initial_ee_orientation = initial_ee_pose[:3, :3]
# Get target end-effector pose (at home position)
data.qpos[:7] = home_pos
mujoco.mj_forward(model, data)
q_home, _ = get_states(data)
home_ee_pose = get_ee_pose(q_home, robot)
home_ee_position = home_ee_pose[:3, 3]
home_ee_orientation = home_ee_pose[:3, :3]
# Create Cartesian trajectory from initial pose to home pose over 2 seconds
num_home_steps = int(2 / dt)
home_cartesian_positions = []
home_cartesian_orientations = []
# Linear interpolation for position
for i in range(num_home_steps):
    alpha = i / (num_home_steps - 1)
    interp_pos = (1 - alpha) * initial_ee_position + alpha * home_ee_position
    home_cartesian_positions.append(interp_pos)
# Spherical interpolation for orientation
home_cartesian_orientations = interpolate_orientations(
    initial_ee_orientation, home_ee_orientation, num_steps=num_home_steps
)
# Compute joint space trajectory for home movement using IK
home_trajectory = []
q_current = q_initial[:6]  # Start from initial joint configuration
for i in range(num_home_steps):
    target_pos = home_cartesian_positions[i]
    target_ori = home_cartesian_orientations[i]
    q_des = inverse_kinematics(q_current, target_pos, target_ori, robot)
    home_trajectory.append(q_des)
    q_current = q_des  # Use previous solution as initial guess for next IK
# Task-space waypoints (starting from home pose)
bottle_pos = np.array([0.42, 0, 0.2])
glass_pos = np.array([0.4, 0.3, 0.2])
waypoints = [
    home_ee_position,  # Start from home position
    bottle_pos + np.array([0, 0, 0.2]),  # hover above bottle
    bottle_pos,  # descend to bottle (GRASP HERE)
    bottle_pos + np.array([0, 0, 0.2]),  # lift bottle
    glass_pos + np.array([0, 0, 0.2]),  # move above glass
    glass_pos,  # align with glass
]
orientations = [
    home_ee_orientation,  # Start from home orientation
    home_ee_orientation,
    home_ee_orientation,
    home_ee_orientation,
    home_ee_orientation,
    home_ee_orientation,
]
# Pour orientation: rotate wrist around x-axis ~ 90 deg
pour_orientation = (
    home_ee_orientation @ R.from_euler("x", -90, degrees=True).as_matrix()
)
waypoints.append(glass_pos)
orientations.append(pour_orientation)
# Return upright
waypoints.append(glass_pos + np.array([0, 0, 0.2]))
orientations.append(home_ee_orientation)
# Interpolate task-space trajectory
task_trajectory = []
task_trajectory_orient = []
for i in range(len(waypoints) - 1):
    p_start, p_end = waypoints[i], waypoints[i + 1]
    R_start, R_end = orientations[i], orientations[i + 1]
    interp_positions = [
        (1 - alpha) * p_start + alpha * p_end for alpha in np.linspace(0, 1, 2000)
    ]
    interp_orientations = interpolate_orientations(R_start, R_end, num_steps=2000)
    task_trajectory.extend(interp_positions)
    task_trajectory_orient.extend(interp_orientations)
# ----------------------
# Main loop - Combined trajectory
# ----------------------
# Reset to initial position
data.qpos[:7] = initial_position
mujoco.mj_forward(model, data)
cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "bottle_grasp")
total_steps = len(home_trajectory) + len(task_trajectory)
for step in range(total_steps):
    q, dq = get_states(data)
    # Save q and dq to CSV
    with open("joint_states.csv", "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(list(q) + list(dq))
    if step < len(home_trajectory):
        # Phase 1: Move to home position using Cartesian interpolation
        data.qpos[:6] = home_trajectory[step]
        data.qpos[6] = -1.5  # Keep gripper open
    else:
        # Phase 2: Task space trajectory
        task_step = step - len(home_trajectory)
        target_pos = task_trajectory[task_step]
        target_ori = task_trajectory_orient[task_step]
        q_des = inverse_kinematics(q, target_pos, target_ori, robot)
        data.qpos[:6] = q_des[:6]
        # Gripper control based on task phase
        if task_step < 4000:
            data.qpos[6] = -1.5  # Open gripper
        else:
            data.qpos[6] = -0.3  # Closed gripper
        # ---- Close gripper when reaching bottle ----
        if task_step == 4000:  # at bottle grasping point
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
# Wait before closing
time.sleep(2)
viewer.close()
