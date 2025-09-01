import mujoco
import mujoco.viewer
import os

# Path to your scene.xml file
XML_PATH = os.path.join(os.path.dirname(__file__), "scene.xml")

def main():
    # Load the model from XML
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    # Create a data object for simulation
    data = mujoco.MjData(model)

    # Launch the viewer for visualization
    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("Press ESC in the viewer window to exit.")
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()

if __name__ == "__main__":
    main()