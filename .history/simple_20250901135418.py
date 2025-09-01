import mujoco
import pinocchio as pin
import numpy as np
import mujoco.viewer as viewer
import time

model = mujoco.MjModel.from_xml_path("./scene.xml")
data = mujoco.MjData(model)
dt = 0.001
home_pos = np.array([0, 0.785, -0.261, -0.523, 0, 0])
data.qpos[:6] = home_pos
mujoco.mj_forward(model, data)
viewer = viewer.launch_passive(model, data)
while True:
    viewer.sync()
    time.sleep(dt)
