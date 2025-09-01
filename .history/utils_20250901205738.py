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
    return data.qpos, data.qvel


def get_ee_pos(body_name="gripper_Mover"):
    return data.get_body_xmat(body_name)
