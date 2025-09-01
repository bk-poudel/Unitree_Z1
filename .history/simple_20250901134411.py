import mujoco
import pinocchio as pin
import numpy as np
import mujoco.viewer as viewer

model = mujoco.MjModel.from_xml_path("path/to/your/scene.xml")
