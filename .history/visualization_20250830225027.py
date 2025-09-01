import mujoco
import mujoco.viewer
import sys


def main():
    # Load the scene.xml file
    model = mujoco.MjModel.from_xml_path("scene.xml")
    data = mujoco.MjData(model)
    # Launch the viewer
    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()
