# Standard library imports
import time

# Third-party library imports
import mujoco
import numpy as np
import mujoco.viewer as viewer
from scipy.spatial.transform import Rotation as R

# Local imports
from utils import *

# ==================================
# 1. SIMULATION SETUP
# ==================================
dt = 0.001  # Simulation timestep (1kHz frequency)
model = mujoco.MjModel.from_xml_path("./scene.xml")
data = mujoco.MjData(model)
# Set up the robot wrapper. This class helps abstract common robot operations.
robot = RobotWrapper.BuildFromURDF("./z1.urdf")
# Set initial joint positions. Home position is all joints at zero.
home_pos = np.array([0, 0.785, -0.261, -0.523, 0, 0])
data.qpos[:6] = home_pos
mujoco.mj_forward(model, data)
# Initialize the MuJoCo viewer.
viewer = viewer.launch_passive(model, data)
# ==================================
# 2. TRAJECTORY DEFINITION
# ==================================
# Define the key points for the task's trajectory.
initial_position = get_ee_pose(home_pos, robot)[:3, 3]
initial_orientation = get_ee_pose(home_pos, robot)[:3, :3]
# Position just above the bottle to avoid collision
bottle_reach_pos = np.array([0.3, 0.0, 0.4])
# Position to grasp the bottle
bottle_grasp_pos = np.array([0.3, 0.0, 0.18])
# Position just above the wine glass
glass_reach_pos = np.array([0.3, 0.4, 0.3])
# Position to release the bottle
glass_release_pos = np.array([0.3, 0.4, 0.18])
# Define the number of points for each linear segment.
num_points = 150
# Create a sequence of trajectories, one for each axis-aligned movement.
# Each movement in the original plan is now broken into three sub-movements (X, Y, Z).
trajectories = []
current_pos = initial_position
# --- Segment 1: Reach the bottle (axis-aligned) ---
# Move in X
x_intermediate = np.array([bottle_reach_pos[0], current_pos[1], current_pos[2]])
trajectories.append(np.linspace(current_pos, x_intermediate, num=num_points))
current_pos = x_intermediate
# Move in Y
y_intermediate = np.array([current_pos[0], bottle_reach_pos[1], current_pos[2]])
trajectories.append(np.linspace(current_pos, y_intermediate, num=num_points))
current_pos = y_intermediate
# Move in Z
trajectories.append(np.linspace(current_pos, bottle_reach_pos, num=num_points))
current_pos = bottle_reach_pos
# --- Segment 2: Grasp the bottle (axis-aligned) ---
# Move in X
x_intermediate = np.array([bottle_grasp_pos[0], current_pos[1], current_pos[2]])
trajectories.append(np.linspace(current_pos, x_intermediate, num=num_points))
current_pos = x_intermediate
# Move in Y
y_intermediate = np.array([current_pos[0], bottle_grasp_pos[1], current_pos[2]])
trajectories.append(np.linspace(current_pos, y_intermediate, num=num_points))
current_pos = y_intermediate
# Move in Z
trajectories.append(np.linspace(current_pos, bottle_grasp_pos, num=num_points))
current_pos = bottle_grasp_pos
# --- Segment 3: Move to the wine glass (axis-aligned) ---
# Move in X
x_intermediate = np.array([glass_release_pos[0], current_pos[1], current_pos[2]])
trajectories.append(np.linspace(current_pos, x_intermediate, num=num_points))
current_pos = x_intermediate
# Move in Y
y_intermediate = np.array([current_pos[0], glass_release_pos[1], current_pos[2]])
trajectories.append(np.linspace(current_pos, y_intermediate, num=num_points))
current_pos = y_intermediate
# Move in Z
trajectories.append(np.linspace(current_pos, glass_release_pos, num=num_points))
current_pos = glass_release_pos
# --- Segment 4: Release the bottle (axis-aligned) ---
# Move in X
x_intermediate = np.array([glass_reach_pos[0], current_pos[1], current_pos[2]])
trajectories.append(np.linspace(current_pos, x_intermediate, num=num_points))
current_pos = x_intermediate
# Move in Y
y_intermediate = np.array([current_pos[0], glass_reach_pos[1], current_pos[2]])
trajectories.append(np.linspace(current_pos, y_intermediate, num=num_points))
current_pos = y_intermediate
# Move in Z
trajectories.append(np.linspace(current_pos, glass_reach_pos, num=num_points))
print_model_info(robot)
# ==================================
# 3. CONTROL LOOP
# ==================================
# Cartesian impedance control parameters
K_cartesian = np.diag([200, 200, 200, 200, 200, 200])  # Stiffness matrix
D_cartesian = np.diag([30, 30, 30, 30, 30, 30])  # Damping matrix
segment_index = 0
point_in_segment_index = 0
total_segments = len(trajectories)
# Main simulation loop
while viewer.is_running():
    # Check if all segments are completed
    if segment_index >= total_segments:
        # If trajectory is finished, hold at the last point of the last segment
        # target_pos = trajectories[-1][-1]
        target_pos = initial_position
    else:
        # Get the current trajectory segment
        current_trajectory = trajectories[segment_index]
        # Check if the current segment is finished
        if point_in_segment_index >= len(current_trajectory):
            segment_index += 1
            point_in_segment_index = 0
            # If we've advanced to a new segment, get its first point.
            if segment_index < total_segments:
                current_trajectory = trajectories[segment_index]
                target_pos = current_trajectory[point_in_segment_index]
            else:
                target_pos = current_trajectory[-1]
        else:
            # Get the target position for the current step
            target_pos = current_trajectory[point_in_segment_index]
    # Get the current robot state (joint angles and velocities)
    q, dq = get_states(data)
    gravity = get_gravity(robot, q)
    # Get the current end-effector pose
    current_ee_pose = get_ee_pose(q, robot)
    # Calculate position error
    position_error = target_pos - current_ee_pose[:3, 3]
    # Orientation control: maintain the initial orientation
    target_orientation = initial_orientation
    error_rot_matrix = target_orientation @ current_ee_pose[:3, :3].T
    error_rotation = R.from_matrix(error_rot_matrix)
    orientation_error = error_rotation.as_rotvec()
    # Combine position and orientation errors
    error = np.concatenate([position_error, orientation_error])
    # Get the end-effector velocity using the Jacobian
    J = get_jacobian(robot, q)
    ee_vel = J @ dq
    # Calculate the desired force in Cartesian space
    force = K_cartesian @ error - D_cartesian @ ee_vel
    # Map the Cartesian force to joint torques using the transpose of the Jacobian
    # torque = J.T @ force + gravity
    # torque = np.zeros(6)
    # Gripper control logic based on the major segments (each has 3 sub-segments)
    gripper_command = 0  # Default to open
    if segment_index < 6:  # First 2 segments (3 sub-segments each)
        gripper_command = -1.5  # Close gripper
    elif segment_index >= 6 and segment_index < 9:  # Third major segment
        gripper_command = -1.5  # Keep gripper closed
    else:
        gripper_command = 0  # Open gripper
    # The gripper command is appended to the main torque array
    # Apply the torques and gripper command to the robot
    send_torques(model, data, torque, viewer)
    # Update index for the next step in the trajectory
    point_in_segment_index += 1
    # Step the simulation
    mujoco.mj_step(model, data)
    # Pause to maintain the correct simulation speed
    time.sleep(dt)
