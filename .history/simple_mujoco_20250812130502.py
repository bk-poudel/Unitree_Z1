#!/usr/bin/env python3
"""
Minimal MuJoCo script to load scene.xml
This is the simplest possible version that loads and runs the scene.
"""

def main():
    import os
    
    # Get scene.xml path
    scene_path = os.path.join(os.path.dirname(__file__), "scene.xml")
    
    if not os.path.exists(scene_path):
        print(f"Error: scene.xml not found at {scene_path}")
        return
    
    try:
        # Import MuJoCo
        import mujoco
        
        # Load model and create data
        print("Loading scene.xml...")
        model = mujoco.MjModel.from_xml_path(scene_path)
        data = mujoco.MjData(model)
        
        print("Model loaded successfully!")
        print(f"Bodies: {model.nbody}, Joints: {model.njnt}, Actuators: {model.nu}")
        
        # Run simulation
        print("Running simulation...")
        for step in range(100):
            mujoco.mj_step(model, data)
            if step % 25 == 0:
                print(f"Step {step}: t={data.time:.2f}s")
        
        print("Simulation complete!")
        
        # Try to open viewer
        try:
            print("Opening viewer (press ESC to close)...")
            with mujoco.viewer.launch_passive(model, data) as viewer:
                while viewer.is_running():
                    mujoco.mj_step(model, data)
                    viewer.sync()
        except Exception as e:
            print(f"Viewer not available: {e}")
            
    except ImportError:
        print("MuJoCo not installed. Run: pip install mujoco")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
