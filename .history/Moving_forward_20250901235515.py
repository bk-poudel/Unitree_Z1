import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R
import mujoco.viewer as viewer
import time
from utils import *

# Simulation parameters
dt = 0.001
model = mujoco.MjModel.from_xml_path("./scene.xml")
data = mujoco.MjData(model)
robot = RobotWrapper.BuildFromURDF(
    "./z1.urdf", "/home/bibek/Unitree_Z1/Unitree_Z1/z1_description/meshes"
)
home_pos = np.array([0, 0.785, -0.261, -0.523, 0, 0, 0])
data.qpos[:7] = home_pos
mujoco.mj_forward(model, data)
q, dq = get_states(data)
viewer = viewer.launch_passive(model, data)
print_model_info(robot)
# Initial and target poses
initial_pose_matrix = get_ee_pose(q, robot)
initial_position = initial_pose_matrix[:3, 3]
initial_orientation = initial_pose_matrix[:3, :3]
target_position = initial_position + np.array([0.1, 0.1, 0.1])
target_orientation = R.from_euler("xyz", [0, 0.5, 0]).as_matrix() @ initial_orientation
# Create a trajectory
num_steps = 1000
position_trajectory = np.linspace(initial_position, target_position, num_steps)
orientation_trajectory = R.from_matrix(initial_orientation).slerp(
    R.from_matrix(target_orientation), np.linspace(0, 1, num_steps)
)
# Impedance control parameters (stiffness and damping)
K_p = np.diag([200, 200, 200])  # Positional stiffness
D_p = np.diag([10, 10, 10])  # Positional damping
K_o = np.diag([50, 50, 50])  # Orientational stiffness
D_o = np.diag([5, 5, 5])  # Orientational damping
# Simulation loop
for i in range(num_steps):
    q, dq = get_states(data)
    # Get current pose
    current_ee_pose = get_ee_pose(q, robot)
    current_pos = current_ee_pose[:3, 3]
    current_rot_matrix = current_ee_pose[:3, :3]
    # Get desired pose from trajectory
    desired_pos = position_trajectory[i]
    desired_rot_matrix = orientation_trajectory[i].as_matrix()
    # Calculate errors
    position_error = desired_pos - current_pos
    # Orientation error is calculated using the axis-angle representation
    delta_rotation = R.from_matrix(desired_rot_matrix @ current_rot_matrix.T)
    orientation_error = delta_rotation.as_rotvec()
    # Get the 6xN full geometric Jacobian
    J_full = get_jacobian(robot, q)
    J_pos = J_full[:3, :]
    J_rot = J_full[3:, :]
    # Compute end-effector velocities (linear and angular)
    ee_linear_vel = J_pos @ dq
    ee_angular_vel = J_rot @ dq
    # Compute desired force and torque
    force = K_p @ position_error - D_p @ ee_linear_vel
    torque = K_o @ orientation_error - D_o @ ee_angular_vel
    # Combine forces and torques into a single wrench vector
    wrench = np.concatenate([force, torque])
    # Map Cartesian wrench to joint torques using the full Jacobian
    tau = J_full.T @ wrench
    # Add gravity compensation (critical for stability)
    tau_grav = data.qfrc_bias[:7]
    tau += tau_grav
    # Apply torques to the robot and step the simulation
    send_torques(model, data, tau, viewer)
    time.sleep(dt)
viewer.close()
