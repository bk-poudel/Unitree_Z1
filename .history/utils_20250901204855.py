import numpy as np
from scipy.spatial.transform import Rotation as R
import mujoco

model = mujoco.MjModel.from_xml_path("./z1.xml")
data = mujoco.MjData(model)
