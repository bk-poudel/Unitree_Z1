import mujoco
import numpy as np
import mujoco.viewer as viewer
import time
from utils import *

# Simulation parameters
dt = 0.001  # 1khz frequency
model = mujoco.MjModel.from_xml_path("./scene.xml")
data = mujoco.MjData(model)
home_pos = np.array([0, 0.785, -0.261, -0.523, 0, 0, 0])
data.qpos[:7] = home_pos
mujoco.mj_forward(model, data)
viewer = viewer.launch_passive(model, data)


def get_states():
    return data.qpos[:7], data.qvel[:7]


print_model_info()
while True:
    q, dq = get_states()
    print(f"q: {q}, dq: {dq}")
    # Step the simulation
    mujoco.mj_forward(model, data)
    print(f"End Effector Pose: {get_ee_pose()} \n")
    print(f"Joint States: {get_states()}\n")
    viewer.sync()
    time.sleep(dt)
