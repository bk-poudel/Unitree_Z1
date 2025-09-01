import mujoco
import numpy as np
import mujoco.viewer as viewer
import time

# Simulation parameters
dt = 0.001  # 1khz frequency
model = mujoco.MjModel.from_xml_path("./z1.xml")
data = mujoco.MjData(model)
viewer = viewer.launch_passive(model, data)
