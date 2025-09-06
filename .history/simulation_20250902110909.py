import mujoco
import numpy as np
import mujoco.viewer as viewer
import time
from utils import *  # Assuming this contains your helper functions
from scipy.spatial.transform import Rotation as R

# Simulation parameters
dt = 0.001  # 1kHz frequency
model = mujoco.MjModel.from_xml_path("./scene.xml")
data = mujoco.MjData(model)
robot = RobotWrapper.BuildFromURDF(
    "./z1.urdf", "/home/bibek/Unitree_Z1/Unitree_Z1/z1_description/meshes"
)
model.opt.gravity[:] = 0.0
home_pos = np.array([0, 0.785, -0.261, -0.523, 0, 0, 0])
data.qpos[:7] = home_pos
mujoco.mj_forward(model, data)
q, dq = get_states(data)
viewer = viewer.launch_passive(model, data)
initial_position = get_ee_pose(q, robot)[:3, 3]
initial_orientation = get_ee_pose(q, robot)[:3, :3]
# Define key points for the trajectory
# Position just above the bottle
bottle_reach_pos = np.array([0.4, 0.0, 0.25])
# Position to grasp the bottle
bottle_grasp_pos = np.array([0.4, 0.0, 0.18])
# Position just above the wine glass
glass_reach_pos = np.array([0.4, 0.4, 0.25])
# Position to release the bottle
glass_release_pos = np.array([0.4, 0.4, 0.18])
# Trajectory segments
num_points = 100
# 1. Reach the bottle
trajectory_1_reach_bottle = np.linspace(
    initial_position, bottle_reach_pos, num=num_points
)
# 2. Grasp the bottle (move down)
trajectory_2_grasp_bottle = np.linspace(
    bottle_reach_pos, bottle_grasp_pos, num=num_points
)
# 3. Move to the wine glass
trajectory_3_move_to_glass = np.linspace(
    bottle_grasp_pos, glass_release_pos, num=num_points
)
# 4. Release the bottle (move up)
trajectory_4_release_bottle = np.linspace(
    glass_release_pos, glass_reach_pos, num=num_points
)
# Concatenate all trajectories
full_trajectory = np.concatenate(
    [
        trajectory_1_reach_bottle,
        trajectory_2_grasp_bottle,
        trajectory_3_move_to_glass,
        trajectory_4_release_bottle,
    ]
)
print_model_info(robot)
# Cartesian impedance control parameters
K_cartesian = np.diag([800, 800, 800, 800, 800, 800])  # Stiffness
D_cartesian = np.diag([100, 100, 100, 100, 100, 100])  # Damping
index = 0
total_steps = len(full_trajectory)
while viewer.is_running():
    if index >= total_steps:
        # Stop or hold at the last point
        target = full_trajectory[-1]
    else:
        target = full_trajectory[index]
    q, dq = get_states(data)
    current_ee_pose = get_ee_pose(q, robot)
    position_error = target - current_ee_pose[:3, 3]
    # Orientation control
    target_orientation = initial_orientation  # Maintain initial orientation
    error_rot_matrix = target_orientation @ current_ee_pose[:3, :3].T
    error_rotation = R.from_matrix(error_rot_matrix)
    orientation_error = error_rotation.as_rotvec()
    error = np.concatenate([position_error, orientation_error])
    # End-effector velocity
    J = get_jacobian(robot, q)
    ee_vel = J @ dq
    # Desired force in Cartesian space
    force = K_cartesian @ error - D_cartesian @ ee_vel
    # Map Cartesian force to joint torques
    torque = J.T @ force
    # Gripper control logic based on trajectory segment
    if index < num_points * 2:
        # Grasp the bottle during the second segment
        gripper_command = -1.5  # Close gripper
    elif index >= num_points * 2 and index < num_points * 3:
        # Hold the bottle during the third segment
        gripper_command = -1.5  # Keep gripper closed
    else:
        # Release the bottle during the fourth segment
        gripper_command = 0  # Open gripper
    send_torques(model, data, torque, viewer, gripper_command)
    # Update index for the next step
    index += 1
    # Step the simulation
    mujoco.mj_step(model, data)
    viewer.sync()
    time.sleep(dt)
