import mujoco
import pinocchio as pin
import numpy as np
import mujoco.viewer as viewer

model = mujoco.MjModel.from_xml_path("path/to/your/scene.xml")
data = mujoco.MjData(model)
dt = 0.001
