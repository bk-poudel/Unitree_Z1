import numpy as np
from scipy.spatial.transform import Rotation as R
import mujoco

model = mujoco.MjModel.from_xml_path("./z1.xml")
data = mujoco.MjData(model)


def print_joint_names():
    for i in range(model.njnt):
        print(f"Joint {i}: {model.joint_names[i]}")


def print_states():
    return data.qpos, data.qvel
