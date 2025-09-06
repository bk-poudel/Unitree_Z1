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
# Gripper control gains
kp_gripper = 1000
kd_gripper = 100


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
# Smooth gripper transition function
# ----------------------
def smooth_gripper_transition(
    data, q_des, start_pos, end_pos, steps, model, viewer, dt
):
    """Smoothly transition gripper position using torque control"""
    for i in range(steps):
        # Keep arm steady with torque control
        q, dq = get_states(data)
        tau_arm = kp * (q_des[:6] - q[:6]) + kd * (np.zeros(6) - dq[:6])
        data.qfrc_applied[:6] = tau_arm
        # Smooth gripper position interpolation
        alpha = i / (steps - 1) if steps > 1 else 0
        # Use cubic interpolation for smoother motion
        alpha_smooth = 3 * alpha**2 - 2 * alpha**3
        gripper_pos = start_pos + alpha_smooth * (end_pos - start_pos)
        # Apply gripper torque control
        data.qpos[6] = tau_gripper
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(dt)


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
# Initialize CSV file
# ----------------------
with open("joint_states.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([f"q{i+1}" for i in range(7)] + [f"dq{i+1}" for i in range(7)])
# ----------------------
# Main loop (torque control with smooth gripper transitions)
# ----------------------
gripper_transition_active = False
step = 0
while step < len(trajectory):
    target_pos = trajectory[step]
    target_ori = trajectory_orient[step]
    q, dq = get_states(data)
    # Save q and dq
    with open("joint_states.csv", "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(list(q) + list(dq))
    # IK for arm
    q_des = inverse_kinematics(q, target_pos, target_ori, robot)
    dq_des = np.zeros(6)
    # Arm torque control
    tau_arm = kp * (q_des[:6] - q[:6]) + kd * (dq_des - dq[:6])
    data.qfrc_applied[:6] = tau_arm
    # Handle special gripper events
    if step == 2000 and not gripper_transition_active:
        print(f"Starting gripper opening at step {step}")
        gripper_transition_active = True
        smooth_gripper_transition(data, q_des, 0.0, -1.5, 200, model, viewer, dt)
        gripper_transition_active = False
        print("Gripper opening completed")
    elif step == 6000 and not gripper_transition_active:
        print(f"Starting gripper closing at step {step}")
        gripper_transition_active = True
        smooth_gripper_transition(data, q_des, -1.5, -0.8, 200, model, viewer, dt)
        gripper_transition_active = False
        print("Gripper closing completed")
    else:
        # Normal operation - maintain current gripper position with position control
        if step < 2000:
            gripper_target = 0.0
        elif step < 6000:
            gripper_target = -1.5
        else:
            gripper_target = -0.8
        data.qpos[6] = gripper_target
        # Normal simulation step
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(dt)
    step += 1
print("Simulation completed")
viewer.close()
