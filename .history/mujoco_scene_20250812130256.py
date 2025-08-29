#!/usr/bin/env python3
"""
Simple MuJoCo script to load and visualize the scene.xml file.
This script demonstrates basic MuJoCo functionality with the Unitree Z1 robot scene.
"""

import mujoco
import mujoco.viewer
import time
import os

def main():
    # Get the current directory where scene.xml is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    scene_path = os.path.join(current_dir, "scene.xml")
    
    # Check if scene.xml exists
    if not os.path.exists(scene_path):
        print(f"Error: scene.xml not found at {scene_path}")
        return
    
    try:
        # Load the MuJoCo model from scene.xml
        print("Loading MuJoCo model from scene.xml...")
        model = mujoco.MjModel.from_xml_path(scene_path)
        data = mujoco.MjData(model)
        
        print(f"Model loaded successfully!")
        print(f"Number of bodies: {model.nbody}")
        print(f"Number of joints: {model.njnt}")
        print(f"Number of actuators: {model.nu}")
        
        # Create a viewer and run the simulation
        print("Starting viewer... (Press ESC to exit)")
        with mujoco.viewer.launch_passive(model, data) as viewer:
            # Run simulation loop
            start_time = time.time()
            while viewer.is_running():
                step_start = time.time()
                
                # Step the simulation
                mujoco.mj_step(model, data)
                
                # Sync viewer with simulation at 60 FPS
                viewer.sync()
                
                # Simple timing control
                time_until_next_step = model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
                    
                # Optional: Add some basic control or animation here
                # For example, you could move joints or apply forces
                
    except Exception as e:
        print(f"Error loading or running simulation: {e}")
        print("Make sure you have MuJoCo properly installed and scene.xml is valid.")

if __name__ == "__main__":
    main()
