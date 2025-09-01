import numpy as np
from scipy.spatial.transform import Rotation as R
import mujoco

# Load the MuJoCo model and data objects.
# These are global to the module so they can be accessed by all functions.
try:
    model = mujoco.MjModel.from_xml_path("./scene.xml")
    data = mujoco.MjData(model)
except FileNotFoundError:
    print("Error: 'z1.xml' not found. Please check the file path.")
    model = None
    data = None


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


def get_ee_pos(body_name="gripperMover"):
    """
    Retrieves the position and rotation matrix of a specified body.
    Args:
        body_name (str): The name of the body (e.g., end-effector).
    Returns:
        tuple: A tuple containing (position, rotation_matrix), or (None, None) if the body is not found.
    """
    if data is None or model is None:
        return None, None
    try:
        # Get the ID of the body from its name
        body_id = model.body(body_name).id
        # The correct way to get the body's position (xpos) and orientation matrix (xmat)
        position = data.body(body_id).xpos
        rotation_matrix = data.body(body_id).xmat
        return position, rotation_matrix
    except KeyError:
        print(f"Error: Body with name '{body_name}' not found in the model.")
        return None, None
