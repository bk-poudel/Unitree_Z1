import mujoco
import numpy as np
import mujoco.viewer as viewer
import time
from utils import *
# Simulation parameters
dt = 0.001  # 1khz frequency
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
wine_bottle_position = np.array([0.4, 0, 0.18])  # Move 10cm forward
target_orientation = initial_orientation
reaching_wine_trajectory = np.linspace(initial_position, wine_bottle_position, num=100)
wine_glass_position = np.array([0.4, 0.5, 0.18])  # Move 20cm forward
wine_glass_reaching_trajectory = np.linspace(
    initial_position, wine_glass_position, num=100
)
wine_bottle_grasping_trajectory = np.linspace(
    wine_bottle_position, wine_bottle_position + np.array([0, 0, 0]), num=100
)
wine_glass_release_trajectory = np.linspace(
    wine_glass_position, wine_glass_position + np.array([0, 0, 0]), num=100
)
trajectory = np.concatenate(
    [
        reaching_wine_trajectory,
        wine_bottle_grasping_trajectory,
        wine_glass_release_trajectory,
    ]
)
print_model_info(robot)
print(get_jacobian(robot, q))
# PD controller gains
# Cartesian impedance control parameters
K_cartesian = np.diag([2000, 2000, 2000, 2000, 2000, 2000])  # Stiffness
D_cartesian = np.diag([400, 400, 400, 400, 400, 400])  # Damping
index = 0
while True:
    if index == len(trajectory):
        index = len(trajectory) - 1
    target = trajectory[index]
    q, dq = get_states(data)
    current_ee_pose = get_ee_pose(q, robot)
    position_error = target - current_ee_pose[:3, 3]
    # Assuming target_orientation and current_orientation are 3x3 rotation matrices
    # from your robot's state.
    # Correct calculation of the error rotation matrix
    error_rot_matrix = target_orientation @ current_ee_pose[:3, :3].T
    # Convert the error matrix to a scipy Rotation object
    error_rotation = R.from_matrix(error_rot_matrix)
    # Calculate the orientation error vector (axis-angle representation)
    orientation_error = error_rotation.as_rotvec()
    error = np.concatenate([position_error, orientation_error])
    # End-effector velocity (approximate using joint velocities and Jacobian)
    J = get_jacobian(robot, q)  # 3x7 Jacobian for position
    J_pos = current_ee_pose[:3, :3] @ J[:3, :]  # Rotate Jacobian to world frame
    J_orient = current_ee_pose[:3, :3] @ J[3:6, :]  # Rotate Jacobian to world frame
    J = np.vstack([J_pos, J_orient])
    ee_vel = J @ dq
    # Desired force in Cartesian space
    force = K_cartesian @ error - D_cartesian @ ee_vel
    # Map Cartesian force to joint torques
    torque = J.T @ force
    # torque = np.array([0, 0, 0, 0, 0, 0])
    # Apply joint-space PD for stability (optional)
    if index <100 or (index>200 and index<300):
    send_torques(model, data, torque, viewer)
    index = index + 1
    time.sleep(dt)
