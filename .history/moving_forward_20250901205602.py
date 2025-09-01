import mujoco
import numpy as np
import mujoco.viewer as viewer
import time
from utils import *

# Simulation parameters
dt = 0.001  # 1khz frequency
model = mujoco.MjModel.from_xml_path("./scene.xml")
data = mujoco.MjData(model)
home_pos = np.array([0, 0.785, -0.261, -0.523, 0, 0])
viewer = viewer.launch_passive(model, data)
print_actuator_names()
print_body_names()
print_joint_names()
while True:
    # Step the simulation
    mujoco.mj_step(model, data)
    viewer.sync()
    time.sleep(dt)
