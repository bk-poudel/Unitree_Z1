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
target_position = np.array([0.3, 0.0, 0.04])  # Move 10cm forward
target_orientation = initial_orientation
trajectory = np.linspace(initial_position, target_position, num=300)
print_model_info(robot)
print(get_jacobian(robot, q))
# PD controller gains
# Cartesian impedance control parameters
K_cartesian = np.diag([150, 150, 150, 150, 150, 150])  # Stiffness
D_cartesian = np.diag([20, 20, 20, 20, 20, 20])  # Damping
steps = 0
joint_limits = []
for j in range(model.njnt):
    if model.jnt_limited[j]:  # check if joint has limits
        joint_limits.append(model.jnt_range[j])
    else:
        joint_limits.append([-np.inf, np.inf])
joint_limits = np.array(joint_limits)
def inverse_kinematics(robot, q_init, pos_des, R_des, joint_limits,
ee_frame="tool0", max_iter=100, tol=1e-4, alpha=0.5):
with joint limits.
del.getFrameId(ee_frame)
es, pos_des)
iter):
nematics(q)
wardKinematics(q)
ot.data.oMf[frame_id]
anslation + rotation log)
_des.inverse() * T_current).vector
rm(err) < tol:
teFrameJacobian(q, frame_id, pin.LOCAL_WORLD_ALIGNED)
.T @ err
joint limits ---
len(q)):
lip(q[i], joint_limits[i, 0], joint_limits[i, 1])
while True:
    if steps == len(trajectory):
        steps = len(trajectory) - 1
    target = trajectory[steps]
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
    # J = get_jacobian(robot, q)  # 3x7 Jacobian for position
    # J_pos = current_ee_pose[:3, :3] @ J[:3, :]  # Rotate Jacobian to world frame
    # J_orient = current_ee_pose[:3, :3] @ J[3:6, :]  # Rotate Jacobian to world frame
    # J = np.vstack([J_pos, J_orient])
    # ee_vel = J @ dq
    # # Desired force in Cartesian space
    # force = K_cartesian @ error - D_cartesian @ ee_vel
    # # Map Cartesian force to joint torques
    # torque = J.T @ force
    # # torque = np.array([0, 0, 0, 0, 0, 0])
    # # Apply joint-space PD for stability (optional)
    # send_torques(model, data, torque, viewer)
    q_des = inverse_kinematics(q, target, target_orientation, robot)
    data.qpos[:6] = q_des
    mujoco.mj_forward(model, data)
    viewer.sync()
    steps = steps + 1
    time.sleep(dt)
