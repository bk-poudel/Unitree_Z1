import mujoco
import mujoco.viewer as mujoco_viewer
import numpy as np
import time
from scipy.spatial.transform import Rotation as R, Slerp
from utils import get_states, get_ee_pose, get_jacobian
from pinocchio import RobotWrapper
import csv
import os

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
# Home position (7 values: 6 arm joints + 1 gripper)
initial_position = np.array([0, 0, 0, 0, 0, 0, 0])
home_pos = np.array([0, 0.785, -0.261, -0.523, 0, 1.57, -1.5])
# Initialize qpos
data.qpos[:7] = initial_position
model.opt.gravity[2] = -9.81
mujoco.mj_forward(model, data)
q, dq = get_states(data)
# Launch viewer (don't overwrite module name)
viewer = mujoco_viewer.launch_passive(model, data)
# Interpolate from initial_position to home_pos over 2 seconds
num_steps = int(2 / dt)
for i in range(num_steps):
    alpha = i / (num_steps - 1)
    interp_q = (1 - alpha) * initial_position + alpha * home_pos
    data.qpos[:7] = interp_q
    mujoco.mj_step(model, data)
    viewer.sync()
    time.sleep(dt)
time.sleep(2)
# ----------------------
# Joint limits from MuJoCo (arm joints only - first 6 qpos assumed to be arm)
# ----------------------
arm_dof = 6
joint_limits_arm = []
# NOTE: This assumes that the first `arm_dof` joints in the MuJoCo XML correspond to the arm
for j in range(arm_dof):
    if j < model.njnt and model.jnt_limited[j]:
        joint_limits_arm.append(model.jnt_range[j].copy())
    else:
        joint_limits_arm.append([-np.inf, np.inf])
joint_limits_arm = np.array(joint_limits_arm)  # shape (6,2)


# ----------------------
# Inverse Kinematics with joint-limit handling (operates on arm_dof joints)
# ----------------------
def inverse_kinematics_arm(
    q_init_full,
    target_pos,
    target_ori,
    robot,
    arm_dof=6,
    joint_limits=joint_limits_arm,
    max_iters=100,
    tol=1e-4,
    alpha=0.5,
):
    """
    q_init_full: full q (including gripper), shape >= arm_dof
    Returns: q_sol_arm (length arm_dof). Caller should preserve gripper separately.
    """
    # Work only with arm joints
    q_current = q_init_full[:arm_dof].copy()
    q_ref = (
        joint_limits[:, 0] + joint_limits[:, 1]
    ) / 2.0  # mid-range reference, length arm_dof
    for it in range(max_iters):
        # Get forward kinematics of the end-effector using Pinocchio (expect 4x4 pose)
        current_ee_pose = get_ee_pose(
            np.concatenate([q_current, q_init_full[arm_dof:]]), robot
        )
        pos_error = target_pos - current_ee_pose[:3, 3]
        R_error = target_ori @ current_ee_pose[:3, :3].T
        ori_error = R.from_matrix(R_error).as_rotvec()
        error = np.concatenate([pos_error, ori_error])
        if np.linalg.norm(error) < tol:
            break
        # Get Jacobian for arm joints only: expect 6 x arm_dof
        J_full = get_jacobian(robot, np.concatenate([q_current, q_init_full[arm_dof:]]))
        # If get_jacobian returns full-J for all joints, slice to arm_dof columns
        if J_full.shape[1] > arm_dof:
            J_full = J_full[:, :arm_dof]
        # Rotate jacobians if your get_jacobian returns them in local frame; here we keep it general.
        # The original code rotated by current_ee_pose[:3,:3]; keep that (maps from local to world)
        J_pos = current_ee_pose[:3, :3] @ J_full[:3, :]
        J_ori = current_ee_pose[:3, :3] @ J_full[3:6, :]
        J6xN = np.vstack([J_pos, J_ori])  # 6 x arm_dof
        # Pseudoinverse
        J_pinv = np.linalg.pinv(J6xN)  # arm_dof x 6
        # Task-space delta
        dq_task = J_pinv @ error  # arm_dof vector
        # Nullspace term: keep near mid-range posture
        N_proj = np.eye(arm_dof) - J_pinv @ J6xN  # arm_dof x arm_dof
        dq_null = N_proj @ (0.1 * (q_ref - q_current))
        dq = dq_task + dq_null
        # simple step-size alpha to help convergence
        q_current = q_current + alpha * dq
        # Clamp to joint limits for arm joints
        for i in range(arm_dof):
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
# Task-space waypoints
# ----------------------
initial_pose = get_ee_pose(q, robot)
initial_position = initial_pose[:3, 3]
initial_orientation = initial_pose[:3, :3]
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
orientations = [initial_orientation] * len(waypoints)
# Pour orientation: rotate wrist around x-axis ~ 90 deg
pour_orientation = (
    initial_orientation @ R.from_euler("x", -90, degrees=True).as_matrix()
)
waypoints.append(glass_pos)
orientations.append(pour_orientation)
waypoints.append(glass_pos + np.array([0, 0, 0.2]))
orientations.append(initial_orientation)
# ----------------------
# Interpolate trajectory
# ----------------------
trajectory = []
trajectory_orient = []
interp_steps_per_segment = 300
for i in range(len(waypoints) - 1):
    p_start, p_end = waypoints[i], waypoints[i + 1]
    R_start, R_end = orientations[i], orientations[i + 1]
    interp_positions = [
        (1 - alpha) * p_start + alpha * p_end
        for alpha in np.linspace(0, 1, interp_steps_per_segment)
    ]
    interp_orientations = interpolate_orientations(
        R_start, R_end, num_steps=interp_steps_per_segment
    )
    trajectory.extend(interp_positions)
    trajectory_orient.extend(interp_orientations)
# ----------------------
# Prepare CSV logging (write header if file doesn't exist)
# ----------------------
csv_path = "joint_states.csv"
if not os.path.exists(csv_path):
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        header = [f"q{i}" for i in range(len(q))] + [f"dq{i}" for i in range(len(dq))]
        writer.writerow(header)
# ----------------------
# Main loop
# ----------------------
# assume 'bottle_grasp' is an equality constraint body name, leaving cid usage as-is
try:
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "bottle_grasp")
except Exception:
    cid = None  # optional - not used further here
for step in range(len(trajectory)):
    target_pos = trajectory[step]
    target_ori = trajectory_orient[step]
    q_full, dq_full = get_states(data)  # full state (7+)
    # Save q and dq to CSV
    with open(csv_path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(list(q_full) + list(dq_full))
    # Solve IK for arm joints only
    q_sol_arm = inverse_kinematics_arm(
        q_full, target_pos, target_ori, robot, arm_dof=arm_dof
    )
    # Keep gripper as current value unless we command it
    gripper_val = q_full[arm_dof] if len(q_full) > arm_dof else -1.5
    # Pre-grasp: keep gripper open until grasp step
    if step < 600:
        gripper_val = -1.5
    else:
        gripper_val = -0.3
    # Set desired arm qpos and gripper
    data.qpos[:arm_dof] = q_sol_arm
    if len(data.qpos) > arm_dof:
        data.qpos[arm_dof] = gripper_val
    # ---- Close gripper when reaching bottle ----
    if step == 600:  # at bottle grasping point
        start = -1.5
        end = -0.3
        close_steps = 200
        for i in range(close_steps):
            # keep arm steady at q_sol_arm
            data.qpos[:arm_dof] = q_sol_arm
            frac = i / (close_steps - 1)
            data.qpos[arm_dof] = start + frac * (end - start)
            mujoco.mj_step(model, data)
            viewer.sync()
            # Sleep but don't expect perfect real-time with dt so small
            time.sleep(dt)
    mujoco.mj_step(model, data)
    viewer.sync()
    time.sleep(dt)
viewer.close()
