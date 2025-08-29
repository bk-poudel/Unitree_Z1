#!/usr/bin/env python3
"""
Simple MuJoCo script to load and visualize scene.xml
This version includes proper error handling and works with modern MuJoCo.
"""

import os
import sys

def check_mujoco_installation():
    """Check if MuJoCo is properly installed."""
    try:
        import mujoco
        return True, mujoco
    except ImportError:
        return False, None

def main():
    # Check MuJoCo installation
    mujoco_available, mujoco = check_mujoco_installation()
    if not mujoco_available:
        print("Error: MuJoCo is not installed!")
        print("\nTo install MuJoCo, run:")
        print("pip install mujoco")
        print("\nFor more information visit:")
        print("https://mujoco.readthedocs.io/en/latest/python.html")
        return
    
    # Get the current directory where scene.xml is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    scene_path = os.path.join(current_dir, "scene.xml")
    
    # Check if scene.xml exists
    if not os.path.exists(scene_path):
        print(f"Error: scene.xml not found at {scene_path}")
        return
    
    print(f"Found scene.xml at: {scene_path}")
    
    try:
        # Load the model
        print("Loading MuJoCo model...")
        
        # Use the proper API calls
        model = mujoco.MjModel.from_xml_path(scene_path)
        data = mujoco.MjData(model)
        
        print("\n" + "="*50)
        print("MODEL LOADED SUCCESSFULLY!")
        print("="*50)
        print(f"Number of bodies: {model.nbody}")
        print(f"Number of joints: {model.njnt}")
        print(f"Number of actuators: {model.nu}")
        print(f"Number of degrees of freedom: {model.nv}")
        print(f"Simulation timestep: {model.opt.timestep}")
        
        # Print body names if available
        print("\nBodies in the model:")
        for i in range(min(10, model.nbody)):  # Show first 10 bodies
            try:
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
                print(f"  {i}: {name if name else 'unnamed'}")
            except:
                print(f"  {i}: body_{i}")
        
        # Run basic simulation
        print("\nRunning basic simulation...")
        for i in range(100):
            mujoco.mj_step(model, data)
            if i % 20 == 0:
                print(f"Step {i}: simulation time = {data.time:.3f}s")
        
        print("\nBasic simulation completed successfully!")
        
        # Try to launch viewer if available
        try:
            import mujoco.viewer
            print("\nStarting interactive viewer...")
            print("Press ESC to exit the viewer.")
            
            with mujoco.viewer.launch_passive(model, data) as viewer:
                while viewer.is_running():
                    mujoco.mj_step(model, data)
                    viewer.sync()
                    
        except ImportError:
            print("\nViewer not available. Install with: pip install mujoco[viewer]")
        except Exception as e:
            print(f"\nViewer error: {e}")
        
    except Exception as e:
        print(f"Error loading or running model: {e}")
        print("\nPossible issues:")
        print("1. scene.xml may have invalid syntax")
        print("2. Referenced files (z1.xml) may be missing")
        print("3. Asset files (.stl) may be missing")

if __name__ == "__main__":
    main()
