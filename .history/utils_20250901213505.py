import numpy as np
from scipy.spatial.transform import Rotation as R
import mujoco
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper

# Load the MuJoCo model and data objects.
# These are global to the module so they can be accessed by all functions.
try:
    model = mujoco.MjModel.from_xml_path("./scene.xml")
    data = mujoco.MjData(model)
    robot = RobotWrapper.BuildFromURDF(
        "./z1.urdf",
        "/home/bibek/Unitree_Z1/Unitree_Z1/z1_description/meshes",
        pin.JointModelFreeFlyer(),
    )
    pin_model = robot.model
    pin_data = robot.data
except FileNotFoundError:
    print("Error: 'z1.xml' not found. Please check the file path.")
    model = None
    data = None


def get_idx_of_ee(name="gripperMover"):
    """
    Returns the index of the end-effector body in the MuJoCo model.
    Args:
        name (str): The name of the end-effector body.
    Returns:
        int: The index of the end-effector body, or -1 if not found.
    """
    if model is None:
        return -1
    try:
        idx = robot.index(name)
        return idx
    except KeyError:
        print(f"Error: Body with name '{name}' not found in the model.")
        return -1


def print_joint_names():
    """
    Prints the names of all joints in the loaded MuJoCo model.
    """
    if model is None:
        return
    for i in range(model.njnt):
        print(f"Joint {i}: {model.joint(i).name}")


def print_body_names():
    """
    Prints the names of all bodies in the loaded MuJoCo model.
    """
    if model is None:
        return
    for i in range(model.nbody):
        print(f"Body {i}: {model.body(i).name}")


def print_actuator_names():
    """
    Prints the names of all actuators in the loaded MuJoCo model.
    """
    if model is None:
        return
    for i in range(model.nu):
        # The correct way to access actuator names in the modern MuJoCo API.
        print(f"Actuator {i}: {model.actuator(i).name}")


def get_states():
    """
    Returns the current joint positions (qpos) and velocities (qvel).
    This function is tailored to return the first 6 elements.
    Returns:
        tuple: A tuple containing (data.qpos, data.qvel).
    """
    if data is None:
        return None, None
    return data.qpos[:6], data.qvel[:6]


def get_ee_pose(site_name="ee_site"):
    """
    Retrieves the position and rotation matrix of a specified site.
    Args:
        site_name (str): The name of the site (e.g., end-effector site).
    Returns:
        tuple: A tuple containing (position, rotation_matrix), or (None, None) if the site is not found.
    """
    if data is None or model is None:
        return None, None
    try:
        # Get the ID of the site from its name
        site_id = model.site(site_name).id
        # Get the site's position and orientation matrix
        position = data.site(site_id).xpos
        rotation_matrix = data.site(site_id).xmat
        return position, rotation_matrix
    except KeyError:
        print(f"Error: Site with name '{site_name}' not found in the model.")
        return None, None
