import mujoco
import os
import mujoco.viewer

# Path to your scene.xml file
xml_path = os.path.join(os.path.dirname(__file__), "scene.xml")
# Load the model
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)
# Launch the viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    print("Press ESC to exit the visualization.")
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
