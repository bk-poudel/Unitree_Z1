import mujoco
import mujoco.viewer
import sys


def main():
    # Load the scene.xml file
    model = mujoco.MjModel.from_xml_path("scene.xml")
    data = mujoco.MjData(model)
    for i in range(model.nbody):
        print(f"Body ID: {i}, Name: {model.body(i).name}")
    # Launch the viewer
    mujoco.viewer.launch(model, data)
    # Print names and ids of all bodies


if __name__ == "__main__":
    main()
