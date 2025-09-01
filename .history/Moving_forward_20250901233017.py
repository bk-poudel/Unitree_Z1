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
home_pos = np.array([0, 0.785, -0.261, -0.523, 0, 0, 0])
data.qpos[:7] = home_pos
mujoco.mj_forward(model, data)
q, dq = get_states(data)
viewer = viewer.launch_passive(model, data)
initial_position = get_ee_pose(q, robot)[:3, 3]
target_position = initial_position + np.array([0.1, 0, 0])  # Move 10cm forward
trajectory = np.linspace(initial_position, target_position, num=1000)
print_model_info(robot)
print(get_jacobian(robot, q))
# PD controller gains
Kp = np.array([50, 50, 50, 50, 50, 50, 50])
Kd = np.array([2, 2, 2, 2, 2, 2, 2])
q_desired = np.copy(home_pos)
for target in trajectory:
    q, dq = get_states(data)
    current_ee_pose = get_ee_pose(q, robot)
    position_error = target - current_ee_pose[:3, 3]
    # Cartesian impedance control parameters
    K_cartesian = np.diag([200, 200, 200])  # Stiffness
    D_cartesian = np.diag([10, 10, 10])  # Damping
    # End-effector velocity (approximate using joint velocities and Jacobian)
    J = get_jacobian(robot, q)  # 3x7 Jacobian for position
    ee_vel = J @ dq[:6]
    # Desired force in Cartesian space
    force = K_cartesian @ position_error - D_cartesian @ ee_vel
    # Map Cartesian force to joint torques
    torque = J.T @ force
    # Apply joint-space PD for stability (optional)
    torque += Kp * (home_pos - q[:6]) - Kd * dq[:6]
    data.ctrl[:7] = torque
    mujoco.mj_step(model, data)
    viewer.sync()
    time.sleep(dt)
