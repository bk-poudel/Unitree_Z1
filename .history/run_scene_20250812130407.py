#!/usr/bin/env python3
"""
Simple MuJoCo script to load and visualize the scene.xml file.
Compatible with different MuJoCo Python binding versions.
"""

import os
import time

def main():
    # Get the current directory where scene.xml is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    scene_path = os.path.join(current_dir, "scene.xml")
    
    # Check if scene.xml exists
    if not os.path.exists(scene_path):
        print(f"Error: scene.xml not found at {scene_path}")
        return
    
    try:
        # Try to import MuJoCo - handle different versions
        try:
            import mujoco
            import mujoco.viewer
            print("Using modern MuJoCo Python bindings...")
            use_modern_api = True
        except ImportError:
            try:
                import mujoco_py as mujoco
                print("Using mujoco-py bindings...")
                use_modern_api = False
            except ImportError:
                print("Error: No MuJoCo Python bindings found!")
                print("Please install either:")
                print("  pip install mujoco  # (recommended, modern)")
                print("  or")
                print("  pip install mujoco-py  # (legacy)")
                return
        
        # Load the model
        print("Loading MuJoCo model from scene.xml...")
        
        if use_modern_api:
            # Modern MuJoCo API
            model = mujoco.MjModel.from_xml_path(scene_path)
            data = mujoco.MjData(model)
            
            print(f"Model loaded successfully!")
            print(f"Number of bodies: {model.nbody}")
            print(f"Number of joints: {model.njnt}")
            print(f"Number of actuators: {model.nu}")
            
            # Create a viewer and run the simulation
            print("Starting viewer... (Press ESC to exit)")
            with mujoco.viewer.launch_passive(model, data) as viewer:
                while viewer.is_running():
                    step_start = time.time()
                    
                    # Step the simulation
                    mujoco.mj_step(model, data)
                    
                    # Sync viewer
                    viewer.sync()
                    
                    # Timing control
                    time_until_next_step = model.opt.timestep - (time.time() - step_start)
                    if time_until_next_step > 0:
                        time.sleep(time_until_next_step)
        else:
            # Legacy mujoco-py API
            model = mujoco.load_model_from_path(scene_path)
            sim = mujoco.MjSim(model)
            viewer = mujoco.MjViewer(sim)
            
            print(f"Model loaded successfully!")
            print(f"Number of bodies: {model.nbody}")
            print(f"Number of joints: {model.njnt}")
            print(f"Number of actuators: {model.nu}")
            
            print("Starting viewer... (Close window to exit)")
            while True:
                sim.step()
                viewer.render()
                
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure MuJoCo is properly installed and scene.xml is valid.")

if __name__ == "__main__":
    main()
