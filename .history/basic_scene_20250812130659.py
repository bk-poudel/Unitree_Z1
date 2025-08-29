#!/usr/bin/env python3
"""
Simple MuJoCo script to load scene.xml
This basic version loads the model and prints information about it.
"""

import os


def main():
    # Get the current directory where scene.xml is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    scene_path = os.path.join(current_dir, "scene.xml")

    # Check if scene.xml exists
    if not os.path.exists(scene_path):
        print(f"Error: scene.xml not found at {scene_path}")
        return

    print(f"Found scene.xml at: {scene_path}")

    try:
        # Try importing mujoco
        import mujoco

        print("MuJoCo imported successfully!")

        # Load the model
        print("Loading MuJoCo model...")
        model = mujoco.MjModel.from_xml_path(scene_path)
        data = mujoco.MjData(model)

        print("\n" + "=" * 50)
        print("MODEL LOADED SUCCESSFULLY!")
        print("=" * 50)
        print(f"Number of bodies: {model.nbody}")
        print(f"Number of joints: {model.njnt}")
        print(f"Number of actuators: {model.nu}")
        print(f"Number of degrees of freedom: {model.nv}")
        print(f"Simulation timestep: {model.opt.timestep}")

        # Try to run a basic simulation
        print("\nRunning basic simulation...")
        for i in range(10):
            mujoco.mj_step(model, data)
            if i % 5 == 0:
                print(f"Step {i}: simulation time = {data.time:.3f}s")

        print("\nSimulation completed successfully!")
        print("\nTo run with visualization, try:")
        print("python run_scene_with_viewer.py")

    except ImportError as e:
        print(f"MuJoCo import error: {e}")
        print("\nTo install MuJoCo, run:")
        print("pip install mujoco")
        print("\nOr visit: https://mujoco.readthedocs.io/en/latest/python.html")

    except Exception as e:
        print(f"Error: {e}")
        print("Make sure scene.xml is a valid MuJoCo model file.")


if __name__ == "__main__":
    main()
