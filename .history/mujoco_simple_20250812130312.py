#!/usr/bin/env python3
"""
Simple MuJoCo script to load scene.xml and run basic simulation without viewer.
Useful for testing and headless environments.
"""

import mujoco
import numpy as np
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
        
        print("="*50)
        print("MODEL INFORMATION")
        print("="*50)
        print(f"Model name: {model.names[0] if model.names else 'Unnamed'}")
        print(f"Number of bodies: {model.nbody}")
        print(f"Number of joints: {model.njnt}")
        print(f"Number of actuators: {model.nu}")
        print(f"Number of degrees of freedom: {model.nv}")
        print(f"Simulation timestep: {model.opt.timestep}")
        
        # Print body names
        print("\nBODIES:")
        for i in range(model.nbody):
            body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
            print(f"  {i}: {body_name if body_name else 'unnamed'}")
        
        # Print joint names
        print("\nJOINTS:")
        for i in range(model.njnt):
            joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
            print(f"  {i}: {joint_name if joint_name else 'unnamed'}")
        
        # Run a few simulation steps
        print("\nRunning simulation for 100 steps...")
        for step in range(100):
            # Reset data if needed (optional)
            if step == 0:
                mujoco.mj_resetData(model, data)
            
            # Step the simulation
            mujoco.mj_step(model, data)
            
            # Print some basic info every 20 steps
            if step % 20 == 0:
                print(f"Step {step}: time = {data.time:.3f}s")
                if model.njnt > 0:
                    print(f"  Joint positions: {data.qpos[:min(3, len(data.qpos))]}")
        
        print("\nSimulation completed successfully!")
        
    except Exception as e:
        print(f"Error loading or running simulation: {e}")
        print("Make sure you have MuJoCo properly installed and scene.xml is valid.")

if __name__ == "__main__":
    main()
