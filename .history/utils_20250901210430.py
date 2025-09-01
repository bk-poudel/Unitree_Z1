import numpy as np
from scipy.spatial.transform import Rotation as R
import mujoco

model = mujoco.MjModel.from_xml_path("./z1.xml")
data = mujoco.MjData(model)


def print_joint_names():
    for i in range(model.njnt):
        print(f"Joint {i}: {model.joint(i).name}")


def print_body_names():
    for i in range(model.nbody):
        print(f"Body {i}: {model.body(i).name}")


def print_actuator_names():
    for i in range(model.nu):
        # Correct way to access actuator names
        print(f"Actuator {i}: {model.actuator(i).name}")


def get_states():
    return data.qpos[:6], data.qvel[:6]


def get_ee_pos(body_name="gripper_Mover"):
    # Get the ID of the body from its name
    body_id = model.body(body_name).id
    # Correct way to get the body's position (xpos) and orientation matrix (xmat)
    position = data.body(body_id).xpos
    rotation_matrix = data.body(body_id).xmat
    return position, rotation_matrix
