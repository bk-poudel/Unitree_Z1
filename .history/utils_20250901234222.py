import numpy as np
from scipy.spatial.transform import Rotation as R
import mujoco
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper

END_EFF_FRAME_ID = 15  # Frame ID for the end-effector in the Pinocchio model


def print_model_info(robot):
    """Prints the names and IDs of all bodies and frames in the model."""
    print("--- Bodies ---")
    # Corrected line: Access names from the underlying model object.
    for i, name in enumerate(robot.model.names):
        print(f"ID: {i}, Name: {name}")
    print("\n--- Frames ---")
    # Corrected line: Access frame names from the underlying model object.
    for i, name in enumerate(robot.model.frames):
        print(f"ID: {i}, Name: {name.name}")


def get_gravity(robot, q):
    """Computes the gravity vector for a given joint configuration."""
    return pin.computeGeneralizedGravity(robot.model, robot.data, q)[:6]


def get_jacobian(robot, q):
    """Computes the Jacobian of the end-effector."""
    pin.computeFrameJacobian(robot.model, robot.data, q, END_EFF_FRAME_ID)
    J_temp = pin.getFrameJacobian(
        robot.model, robot.data, END_EFF_FRAME_ID, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
    )
    J = np.zeros([6, robot.model.nv])
    J[3:6, :] = J_temp[0:3, :]
    J[0:3, :] = J_temp[3:6, :]
    return J[:, : robot.model.nv]


def get_idx_of_ee(pin_model, name="gripperMover"):
    """
    Returns the index of the end-effector body in the MuJoCo model.
    Args:
        name (str): The name of the end-effector body.
    Returns:
        int: The index of the end-effector body, or -1 if not found.
    """
    if pin_model is None:
        return -1
    try:
        # Corrected line: Use pin_model.getJointId to find the joint index.
        idx = pin_model.getJointId(name)
        return idx
    except Exception:
        print(f"Error: Body with name '{name}' not found in the model.")
        return -1


def print_joint_names(model):
    """
    Prints the names of all joints in the loaded MuJoCo model.
    """
    if model is None:
        return
    for i in range(model.njnt):
        print(f"Joint {i}: {model.joint(i).name}")


def print_body_names(model):
    """
    Prints the names of all bodies in the loaded MuJoCo model.
    """
    if model is None:
        return
    for i in range(model.nbody):
        print(f"Body {i}: {model.body(i).name}")


def print_actuator_names(model):
    """

    Prints the names of all actuators in the loaded MuJoJo model.
    """
    if model is None:
        return
    for i in range(model.nu):
        # The correct way to access actuator names in the modern MuJoCo API.
        print(f"Actuator {i}: {model.actuator(i).name}")


def get_states(model, data):
    """
    Returns the current joint positions (qpos) and velocities (qvel).
    This function is tailored to return the first nv elements.
    Returns:
        tuple: A tuple containing (data.qpos, data.qvel).
    """
    if data is None:
        return None, None
    return data.qpos[: model.nv], data.qvel[: model.nv]


def get_ee_pose(q, robot):
    """
    Retrieves the position and rotation matrix of a specified site.
    Args:
        site_name (str): The name of the site (e.g., end-effector site).
    Returns:
        tuple: A tuple containing (position, rotation_matrix), or (None, None) if the site is not found.
    """
    robot.forwardKinematics(q)
    T_S_F = robot.framePlacement(q, END_EFF_FRAME_ID)
    return T_S_F.homogeneous


def send_torques(model, data, tau, viewer):
    data.ctrl[: tau.shape[0]] = tau
    mujoco.mj_step(model, data)
    if viewer:
        viewer.sync()
