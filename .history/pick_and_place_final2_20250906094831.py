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
# Load robot URDF in Pinocchio (adjust mesh path as needed)
robot = RobotWrapper.BuildFromURDF(
    "./z1.urdf", "/home/bibek/Unitree_Z1/Unitree_Z1/z1_description/meshes"
)
# Home position (7 joints: 6 arm + 1 gripper)
initial_position = np.array([0, 0, 0, 0, 0, 0, 0])  # set gripper open initially (-1.5)
home_pos = np.array([0, 0.785, -0.261, -0.523, 0, 1.57, -1.5])
# initialize model
data.qpos[:7] = initial_position
model.opt.gravity[2] = -9.81
mujoco.mj_forward(model, data)
# helper: number of arm joints (6) and the gripper index
ARM_DOF = 6
GRIPPER_IDX = 6
# Launch viewer (passive)
viewer = viewer.launch_passive(model, data)
# ----------------------
# Joint limits from MuJoCo (for arm only)
# ----------------------
joint_limits = []
for j in range(model.njnt):
    if model.jnt_limited[j]:
        joint_limits.append(model.jnt_range[j])
    else:
        joint_limits.append([-np.inf, np.inf])
joint_limits = np.array(joint_limits)
joint_limits_arm = joint_limits[:ARM_DOF]  # only arm joints


# ----------------------
# Inverse Kinematics (operates on 6-DOF arm only)
# ----------------------
def inverse_kinematics_arm(
    q_init_arm, target_pos, target_ori, robot, max_iters=100, tol=1e-4
):
    """
    q_init_arm: length 6 array for arm joints
    returns: length-6 q_arm solution
    """
    q_current = q_init_arm.copy()
    q_ref = (
        joint_limits_arm[:, 0] + joint_limits_arm[:, 1]
    ) / 2.0  # mid-range reference
    for _ in range(max_iters):
        current_ee_pose = get_ee_pose(q_current, robot)  # full 4x4 pose
        pos_err = target_pos - current_ee_pose[:3, 3]
        R_error = target_ori @ current_ee_pose[:3, :3].T
        ori_err = R.from_matrix(R_error).as_rotvec()
        error = np.concatenate([pos_err, ori_err])
        if np.linalg.norm(error) < tol:
            break
        J = get_jacobian(robot, q_current)  # should be 6 x 6 for a 6-DOF arm
        # Ensure J is shaped properly
        J_full = J  # using 6x6 directly (pos+ori)
        # damped least squares (more stable than plain pinv)
        lam = 1e-3
        JtJ = J_full.T @ J_full + lam * np.eye(ARM_DOF)
        dq_task = np.linalg.solve(JtJ, J_full.T @ error)
        # nullspace to stay near q_ref
        N = np.eye(ARM_DOF) - np.linalg.pinv(J_full) @ J_full
        dq_null = 0.1 * N @ (q_ref - q_current)
        dq = dq_task + dq_null
        q_current = q_current + dq
        # clamp to arm joint limits
        for i in range(ARM_DOF):
            q_current[i] = np.clip(
                q_current[i], joint_limits_arm[i, 0], joint_limits_arm[i, 1]
            )
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
# Prepare CSV (header)
# ----------------------
with open("joint_states.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    header = []
    for i in range(7):
        header.append(f"q{i}")
    for i in range(7):
        header.append(f"dq{i}")
    writer.writerow(header)
# ----------------------
# Get current & home EE poses
# ----------------------
# initial -> set and forward
data.qpos[:7] = initial_position
mujoco.mj_forward(model, data)
q_initial, dq_initial = get_states(data)
q_initial = q_initial[:7]
initial_ee_pose = get_ee_pose(q_initial[:ARM_DOF], robot)
initial_ee_position = initial_ee_pose[:3, 3]
initial_ee_orientation = initial_ee_pose[:3, :3]
# home -> set and forward
data.qpos[:7] = home_pos
mujoco.mj_forward(model, data)
q_home, dq_home = get_states(data)
q_home = q_home[:7]
home_ee_pose = get_ee_pose(q_home[:ARM_DOF], robot)
home_ee_position = home_ee_pose[:3, 3]
home_ee_orientation = home_ee_pose[:3, :3]
# ----------------------
# Create Cartesian trajectory for home movement (and gripper interpolation)
# ----------------------
home_duration = 2.0  # seconds
num_home_steps = int(home_duration / dt)
home_cartesian_positions = []
home_cartesian_orientations = []
home_gripper_values = []
# Linear interpolation for position & gripper
for i in range(num_home_steps):
    alpha = i / (num_home_steps - 1)
    interp_pos = (1 - alpha) * initial_ee_position + alpha * home_ee_position
    home_cartesian_positions.append(interp_pos)
# Spherical interpolation for orientation
home_cartesian_orientations = interpolate_orientations(
    initial_ee_orientation, home_ee_orientation, num_steps=num_home_steps
)
# gripper interpolation (linear from initial to home)
g_start = initial_position[GRIPPER_IDX]
g_end = home_pos[GRIPPER_IDX]
for i in range(num_home_steps):
    alpha = i / (num_home_steps - 1)
    g_val = (1 - alpha) * g_start + alpha * g_end
    home_gripper_values.append(g_val)
# Compute joint-space IK for each home Cartesian step (arm only)
home_trajectory_arm = []
q_current_arm = q_initial[:ARM_DOF].copy()
for i in range(num_home_steps):
    target_pos = home_cartesian_positions[i]
    target_ori = home_cartesian_orientations[i]
    q_des_arm = inverse_kinematics_arm(q_current_arm, target_pos, target_ori, robot)
    home_trajectory_arm.append(q_des_arm.copy())
    q_current_arm = q_des_arm
# ----------------------
# Task-space waypoints and orientations (with gripper schedule)
# ----------------------
bottle_pos = np.array([0.42, 0.0, 0.2])
glass_pos = np.array([0.4, 0.3, 0.2])
waypoints = [
    home_ee_position,  # 0 start at home
    bottle_pos + np.array([0, 0, 0.2]),  # 1 hover above bottle
    bottle_pos,  # 2 descend to bottle (GRASP here)
    bottle_pos + np.array([0, 0, 0.2]),  # 3 lift bottle
    glass_pos + np.array([0, 0, 0.2]),  # 4 move above glass
    glass_pos,  # 5 align with glass
]
orientations = [home_ee_orientation] * len(waypoints)
# pour orientation (rotate wrist around x-axis ~90 deg)
pour_orientation = (
    home_ee_orientation @ R.from_euler("x", -90, degrees=True).as_matrix()
)
# add pouring and return upright segments
waypoints.append(glass_pos)  # 6 (start pour)
orientations.append(pour_orientation)
waypoints.append(glass_pos + np.array([0, 0, 0.2]))  # 7 (return upright)
orientations.append(home_ee_orientation)
# For gripper: define gripper target at each waypoint
# default open = -1.5, closed = -0.3
g_open = -1.5
g_closed = -0.3
waypoint_gripper_targets = [
    g_open,  # 0 at home
    g_open,  # 1 hover above bottle
    g_closed,  # 2 at bottle (should end closed)
    g_closed,  # 3 holding bottle
    g_closed,  # 4 above glass
    g_closed,  # 5 align with glass (still holding)
    g_closed,  # 6 pouring
    g_open,  # 7 return upright (release)
]
# Build task trajectory: for each segment, interpolate positions/orientations/gripper
SEG_POINTS = 2000  # points per segment (as you used)
task_trajectory_positions = []
task_trajectory_orientations = []
task_gripper_values = []
for i in range(len(waypoints) - 1):
    p_start = waypoints[i]
    p_end = waypoints[i + 1]
    R_start = orientations[i]
    R_end = orientations[i + 1]
    g_start = waypoint_gripper_targets[i]
    g_end = waypoint_gripper_targets[i + 1]
    # positions
    for alpha in np.linspace(0, 1, SEG_POINTS, endpoint=True):
        task_trajectory_positions.append((1 - alpha) * p_start + alpha * p_end)
        task_gripper_values.append((1 - alpha) * g_start + alpha * g_end)
    # orientations
    interp_orients = interpolate_orientations(R_start, R_end, num_steps=SEG_POINTS)
    task_trajectory_orientations.extend(interp_orients)
# Total lengths
total_home_steps = len(home_trajectory_arm)
total_task_steps = len(task_trajectory_positions)
total_steps = total_home_steps + total_task_steps
# ----------------------
# Reset to initial position and run combined loop
# ----------------------
data.qpos[:7] = initial_position
mujoco.mj_forward(model, data)
# For IK during task portion: we'll use current arm q taken from the simulator state
for step in range(total_steps):
    q_sim, dq_sim = get_states(data)
    q_sim = q_sim[:7]
    dq_sim = dq_sim[:7]
    # log to CSV
    with open("joint_states.csv", "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(list(q_sim) + list(dq_sim))
    if step < total_home_steps:
        # Home phase: use precomputed arm IK + precomputed gripper interpolation
        q_arm_des = home_trajectory_arm[step]
        g_des = home_gripper_values[step]
        # write into MuJoCo qpos
        data.qpos[:ARM_DOF] = q_arm_des
        data.qpos[GRIPPER_IDX] = g_des
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(dt)
    else:
        # Task phase: compute IK for the arm given the current simulator arm state as seed
        task_step = step - total_home_steps
        target_pos = task_trajectory_positions[task_step]
        target_ori = task_trajectory_orientations[task_step]
        g_des = task_gripper_values[task_step]
        q_seed = q_sim[:ARM_DOF].copy()
        q_arm_des = inverse_kinematics_arm(q_seed, target_pos, target_ori, robot)
        data.qpos[:ARM_DOF] = q_arm_des
        data.qpos[GRIPPER_IDX] = g_des
        # If you want to add an explicit gradual gripper closure at the bottle grasp segment,
        # it's already handled by interpolation above (task_gripper_values).
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(dt)
# small pause and close viewer
time.sleep(2)
viewer.close()
