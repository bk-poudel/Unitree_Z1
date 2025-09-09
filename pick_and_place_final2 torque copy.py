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
dt = 0.002  # simulation timestep
model = mujoco.MjModel.from_xml_path("./scene.xml")
data = mujoco.MjData(model)
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
# Control gains (tuned for smoother motion)
# ----------------------
# Arm torque control gains
kp_arm = np.array([1000, 1000, 1000, 1000, 1000, 1000])  # Reduced kp
kd_arm = np.array([100, 100, 80, 50, 25, 25])  # Reduced kd
# Gripper torque control gains
kp_gripper = 200.0  # Reduced kp
kd_gripper = 20.0  # Reduced kd


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
# Generate smooth position trajectory using cubic interpolation
# ----------------------
def generate_position_trajectory(start_pos, end_pos, num_steps):
    """Generate smooth position trajectory using cubic interpolation"""
    trajectory = []
    for i in range(num_steps):
        alpha = i / (num_steps - 1) if num_steps > 1 else 0
        # Cubic interpolation for smoother motion
        alpha_smooth = 3 * alpha**2 - 2 * alpha**3
        pos = start_pos + alpha_smooth * (end_pos - start_pos)
        trajectory.append(pos)
    return trajectory


# ----------------------
# Generate smooth gripper trajectory
# ----------------------
def generate_gripper_trajectory(start_pos, end_pos, num_steps):
    """Generate smooth gripper position trajectory using cubic interpolation"""
    trajectory = []
    for i in range(num_steps):
        alpha = i / (num_steps - 1) if num_steps > 1 else 0
        # Cubic interpolation for smoother motion
        alpha_smooth = 3 * alpha**2 - 2 * alpha**3
        pos = start_pos + alpha_smooth * (end_pos - start_pos)
        trajectory.append(pos)
    return trajectory


# ----------------------
# Task-space waypoints
# ----------------------
q, dq = get_states(data)
initial_pose = get_ee_pose(q, robot)
initial_position = initial_pose[:3, 3]
initial_orientation = initial_pose[:3, :3]
bottle_pos = np.array([0.42, 0, 0.2])
glass_pos = np.array([0.4, 0.3, 0.2])
# Main waypoints for arm movement
main_waypoints = [
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
main_orientations = [
    initial_orientation,
    initial_orientation,
    rotated_orientation,
    rotated_orientation,
    rotated_orientation,
    rotated_orientation,
    rotated_orientation,
]
# Add waypoints for pouring action
pour_orientation = initial_orientation @ R.from_euler("x", 0, degrees=True).as_matrix()
main_waypoints.append(glass_pos)
main_orientations.append(pour_orientation)
main_waypoints.append(glass_pos + np.array([0, 0, 0.2]))
main_orientations.append(rotated_orientation)
# ----------------------
# Build complete trajectory with gripper actions
# ----------------------
trajectory = []
trajectory_orient = []
gripper_trajectory = []
steps_per_segment = 2000
gripper_transition_steps = 200
hold_steps = 500  # Added steps to hold position
# Current gripper position
current_gripper_pos = 0.0
for i in range(len(main_waypoints) - 1):
    p_start, p_end = main_waypoints[i], main_waypoints[i + 1]
    R_start, R_end = main_orientations[i], main_orientations[i + 1]
    # Regular arm movement
    interp_positions = generate_position_trajectory(p_start, p_end, steps_per_segment)
    interp_orientations = interpolate_orientations(
        R_start, R_end, num_steps=steps_per_segment
    )
    trajectory.extend(interp_positions)
    trajectory_orient.extend(interp_orientations)
    # Gripper stays at current position during arm movement
    gripper_trajectory.extend([current_gripper_pos] * steps_per_segment)
    # Add gripper action after specific segments
    if i == 1:  # After reaching above bottle - open gripper
        print(f"Adding gripper opening sequence after segment {i}")
        # Hold position while opening gripper
        trajectory.extend([p_end] * hold_steps)
        trajectory_orient.extend([R_end] * hold_steps)
        gripper_trajectory.extend(
            [current_gripper_pos] * hold_steps
        )  # Keep gripper constant during hold
        # Then, move gripper
        stay_positions = [p_end] * gripper_transition_steps
        stay_orientations = [R_end] * gripper_transition_steps
        gripper_open_traj = generate_gripper_trajectory(
            current_gripper_pos, -1.5, gripper_transition_steps
        )
        trajectory.extend(stay_positions)
        trajectory_orient.extend(stay_orientations)
        gripper_trajectory.extend(gripper_open_traj)
        current_gripper_pos = -1.5
    elif i == 2:  # After reaching above glass - close gripper
        print(f"Adding gripper closing sequence after segment {i}")
        # Hold position while closing gripper
        trajectory.extend([p_end] * hold_steps)
        trajectory_orient.extend([R_end] * hold_steps)
        gripper_trajectory.extend(
            [current_gripper_pos] * hold_steps
        )  # Keep gripper constant during hold
        # Then, move gripper
        stay_positions = [p_end] * gripper_transition_steps
        stay_orientations = [R_end] * gripper_transition_steps
        gripper_close_traj = generate_gripper_trajectory(
            current_gripper_pos, -0.8, gripper_transition_steps
        )
        trajectory.extend(stay_positions)
        trajectory_orient.extend(stay_orientations)
        gripper_trajectory.extend(gripper_close_traj)
        current_gripper_pos = -0.8
print(f"Total trajectory length: {len(trajectory)}")
print(f"Gripper trajectory length: {len(gripper_trajectory)}")
# ----------------------
# Initialize CSV file
# ----------------------
with open("joint_states.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([f"q{i+1}" for i in range(7)] + [f"dq{i+1}" for i in range(7)])
# ----------------------
# Main control loop (both arm and gripper use torque control)
# ----------------------
step = 0
while step < len(trajectory):
    target_pos = trajectory[step]
    target_ori = trajectory_orient[step]
    target_gripper_pos = gripper_trajectory[step]
    q, dq = get_states(data)
    # Save joint states to CSV
    with open("joint_states.csv", "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(list(q) + list(dq))
    # Inverse kinematics for arm
    q_des = inverse_kinematics(q, target_pos, target_ori, robot)
    dq_des = np.zeros(6)
    # Arm torque control
    tau_arm = kp_arm * (q_des[:6] - q[:6]) + kd_arm * (dq_des - dq[:6])
    data.qfrc_applied[:6] = tau_arm
    # Gripper torque control (smooth velocity-based control)
    gripper_error = target_gripper_pos - data.qpos[6]
    gripper_velocity_error = 0.0 - data.qvel[6]  # Target velocity is 0
    tau_gripper = kp_gripper * gripper_error + kd_gripper * gripper_velocity_error
    # Apply gripper torque
    data.qfrc_applied[6] = tau_gripper
    # Simulation step
    mujoco.mj_step(model, data)
    viewer.sync()
    time.sleep(dt)
    step += 1
    # Print progress at key points
    if step % 1000 == 0:
        print(
            f"Step {step}, Target gripper pos: {target_gripper_pos:.3f}, Actual: {data.qpos[6]:.3f}"
        )
print("Simulation completed")
viewer.close()
