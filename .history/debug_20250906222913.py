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
# Torque control gains (for arm only)
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
# Task-space waypoints with holding positions
# ----------------------
q, dq = get_states(data)
initial_pose = get_ee_pose(q, robot)
initial_position = initial_pose[:3, 3]
initial_orientation = initial_pose[:3, :3]
bottle_pos = np.array([0.42, 0, 0.2])
glass_pos = np.array([0.4, 0.3, 0.2])
# Define waypoints with holding positions and pouring
waypoints = [
    # Phase 1: Move to initial position and hold
    initial_position,  # 0: Start position
    initial_position + np.array([0, 0, 0.2]),  # 1: Move up (gripper opens here)
    # Phase 2: Move to bottle and hold before grasping
    bottle_pos + np.array([0, 0, 0.2]),  # 2: Above bottle
    bottle_pos,  # 3: At bottle
    # 4: Hold at bottle (same position for holding)
    bottle_pos,  # 5: Still holding at bottle (gripper closes here)
    # Phase 3: Lift bottle and move to glass
    bottle_pos + np.array([0, 0, 0.2]),  # 6: Lift bottle
    glass_pos + np.array([0, 0, 0.2]),  # 7: Above glass
    glass_pos,  # 8: At glass
    # Phase 4: Pour and lift back up
    glass_pos,  # 10: Continue pouring
    glass_pos + np.array([0, 0, 0.2]),  # 11: Lift up after pouring
]
# Define orientations for each waypoint
rotated_orientation = (
    initial_orientation @ R.from_euler("x", 90, degrees=True).as_matrix()
)
pour_orientation = initial_orientation @ R.from_euler("x", 0, degrees=True).as_matrix()
orientations = [
    # Phase 1: Initial orientations
    initial_orientation,  # 0: Start
    initial_orientation,  # 1: Move up (gripper opens)
    # Phase 2: Approach and grasp bottle
    rotated_orientation,  # 2: Above bottle
    rotated_orientation,  # 3: At bottle
    # rotated_orientation,  # 4: Hold at bottle
    rotated_orientation,  # 5: Still holding (gripper closes)
    # Phase 3: Move with bottle
    rotated_orientation,  # 6: Lift bottle
    rotated_orientation,  # 7: Above glass
    rotated_orientation,  # 8: At glass (still rotated)
    # Phase 4: Pour and finish
    # pour_orientation,  # 9: Hold at glass (change to pour orientation)
    pour_orientation,  # 10: Continue pouring
    rotated_orientation,  # 11: Lift up (back to rotated)
]
# ----------------------
# Interpolate trajectory with different step counts for different phases
# ----------------------
trajectory = []
trajectory_orient = []
step_counts = [
    2000,
    2000,
    2000,
    1000,
    1000,
    2000,
    2000,
    1000,
    1000,
    2000,
]  # Steps for each segment
for i in range(len(waypoints) - 1):
    p_start, p_end = waypoints[i], waypoints[i + 1]
    R_start, R_end = orientations[i], orientations[i + 1]
    num_steps = step_counts[i]
    interp_positions = [
        (1 - alpha) * p_start + alpha * p_end for alpha in np.linspace(0, 1, num_steps)
    ]
    interp_orientations = interpolate_orientations(R_start, R_end, num_steps=num_steps)
    trajectory.extend(interp_positions)
    trajectory_orient.extend(interp_orientations)
# ----------------------
# Initialize CSV file
# ----------------------
with open("joint_states.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([f"q{i+1}" for i in range(7)] + [f"dq{i+1}" for i in range(7)])
# ----------------------
# Calculate key step positions for gripper control
# ----------------------
# Step positions based on cumulative trajectory lengths
step_positions = [0]
for count in step_counts:
    step_positions.append(step_positions[-1] + count)
gripper_open_step = step_positions[
    1
]  # When moving up from initial position (0.2m above)
gripper_close_step = step_positions[4]  # After holding at bottle
print(f"Gripper will open at step: {gripper_open_step}")
print(f"Gripper will close at step: {gripper_close_step}")
print(f"Total trajectory steps: {len(trajectory)}")
# ----------------------
# Main loop (arm uses torque control, gripper uses smooth position control)
# ----------------------
step = 0
while step < len(trajectory):
    target_pos = trajectory[step]
    target_ori = trajectory_orient[step]
    q, dq = get_states(data)
    # Save joint states to CSV
    with open("joint_states.csv", "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(list(q) + list(dq))
    # Inverse kinematics for arm
    q_des = inverse_kinematics(q, target_pos, target_ori, robot)
    dq_des = np.zeros(6)
    # Arm torque control
    tau_arm = kp * (q_des[:6] - q[:6]) + kd * (dq_des - dq[:6])
    data.qfrc_applied[:6] = tau_arm
    # Handle gripper control with smooth continuous transitions
    if step >= gripper_open_step - 100 and step <= gripper_open_step + 100:
        # Smooth transition for gripper opening
        if step <= gripper_open_step:
            progress = (step - (gripper_open_step - 100)) / 100.0
        else:
            progress = 1.0 - (step - gripper_open_step) / 100.0
            progress = max(0, progress) + 1.0
        progress = np.clip(progress, 0, 1)
        # Cubic interpolation for smooth motion
        alpha_smooth = 3 * progress**2 - 2 * progress**3
        gripper_target = 0.0 + alpha_smooth * (-1.5)
    elif step >= gripper_close_step - 100 and step <= gripper_close_step + 100:
        # Smooth transition for gripper closing
        if step <= gripper_close_step:
            progress = (step - (gripper_close_step - 100)) / 100.0
        else:
            progress = 1.0 - (step - gripper_close_step) / 100.0
            progress = max(0, progress) + 1.0
        progress = np.clip(progress, 0, 1)
        # Cubic interpolation for smooth motion
        alpha_smooth = 3 * progress**2 - 2 * progress**3
        gripper_target = -1.5 + alpha_smooth * (-0.6 + 1.5)
    else:
        # Normal operation - set gripper position based on phase
        if step < gripper_open_step - 100:
            gripper_target = 0.0
        elif step < gripper_close_step - 100:
            gripper_target = -1.5
        else:
            gripper_target = -0.6
    data.qpos[6] = gripper_target
    # Normal simulation step
    mujoco.mj_step(model, data)
    viewer.sync()
    time.sleep(dt)
    step += 1
print("Simulation completed")
viewer.close()
