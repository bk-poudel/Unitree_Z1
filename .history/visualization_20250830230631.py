import mujoco
import mujoco.viewer
import sys
import numpy as np


def control_callback(model, data):
    # Apply 0.5N torque to a specific actuator/joint
    # For this example, we'll apply it to the first actuator
    # You may need to adjust the actuator index based on your model
    if model.nu > 0:  # Check if there are any actuators
        data.ctrl[0] = 0.5  # Apply 0.5N torque to the first actuator
    else:
        # If no actuators available, directly apply torque to a specific joint
        # Identify a joint in your model and apply torque directly
        # For example, to apply to the first joint:
        if model.njnt > 0:
            data.qfrc_applied[0] = 0.5


def main():
    # Load the scene.xml file
    model = mujoco.MjModel.from_xml_path("scene.xml")
    data = mujoco.MjData(model)
    # Print information about bodies, actuators and joints
    print(f"Number of bodies: {model.nbody}")
    for i in range(model.nbody):
        print(f"Body ID: {i}, Name: {model.body(i).name}")
    print(f"\nNumber of actuators: {model.nu}")
    for i in range(model.nu):
        print(f"Actuator ID: {i}, Name: {model.actuator(i).name}")
    print(f"\nNumber of joints: {model.njnt}")
    for i in range(model.njnt):
        print(f"Joint ID: {i}, Name: {model.joint(i).name}")
    # Launch the viewer with the custom control callback
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Set the control callback
        control_callback
        # Simulation loop
        while viewer.is_running():
            # Step the simulation
            mujoco.mj_step(model, data)
            # Update the viewer
            viewer.sync()


if __name__ == "__main__":
    main()
