import mujoco
import os
import mujoco.viewer
import numpy as np

# Path to your scene.xml file
xml_path = os.path.join(os.path.dirname(__file__), "scene.xml")
# Load the model
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)
# Launch the viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    print("Press ESC to exit the visualization.")
    while viewer.is_running():
        data.qpos[:6] = np.array([0, 1.5, -1.0, -0.54, 0, 0])
        data.qpos[6] = -1.5
        mujoco.mj_step(model, data)
        viewer.sync()
