import mujoco
import mujoco.viewer
import numpy as np
import time


def main():
    # Load the scene.xml file
    model = mujoco.MjModel.from_xml_path("scene.xml")
    data = mujoco.MjData(model)
    
    # Find relevant actuators and bodies
    z1_base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "z1_base_link")
    bottle_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "waterbottle")
    
    # Find actuator IDs
    gripper_actuator_ids = []
    for i in range(model.nu):
        name = model.actuator(i).name
        if "finger" in name:
            gripper_actuator_ids.append(i)
    
    # Create viewer but don't launch yet
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Reset simulation
        mujoco.mj_resetData(model, data)
        
        # Main simulation loop
        step_count = 0
        grasp_phase = "approach"  # phases: approach, pre_grasp, grasp, lift
        
        while viewer.is_running():
            step_start = time.time()
            
            # Get current positions
            z1_pos = data.xpos[z1_base_id]
            bottle_pos = data.xpos[bottle_id]
            
            # Vector from z1 to bottle
            direction = bottle_pos - z1_pos
            distance = np.linalg.norm(direction[:2])  # XY distance
            
            # State machine for grasping
            if grasp_phase == "approach" and step_count > 50:
                # Apply force to move toward bottle
                if distance > 0.03:
                    # Normalize direction vector (XY only)
                    move_dir = direction[:2] / max(0.001, np.linalg.norm(direction[:2]))
                    # Apply force to move toward bottle
                    data.xfrc_applied[z1_base_id, :2] = move_dir * 1.0
                else:
                    # Close enough to grasp
                    data.xfrc_applied[z1_base_id, :] = 0
                    grasp_phase = "pre_grasp"
                    print("Ready to grasp")
            
            elif grasp_phase == "pre_grasp" and step_count > 150:
                # Align gripper with bottle
                data.xfrc_applied[z1_base_id, :] = 0
                grasp_phase = "grasp"
                print("Grasping")
            
            elif grasp_phase == "grasp" and step_count > 250:
                # Close the gripper
                for actuator_id in gripper_actuator_ids:
                    data.ctrl[actuator_id] = -1.0  # Close gripper (adjust value as needed)
                
                grasp_phase = "lift"
                print("Lifting")
            
            elif grasp_phase == "lift" and step_count > 350:
                # Apply upward force to lift
                data.xfrc_applied[z1_base_id, 2] = 1.0
            
            # Step the simulation
            mujoco.mj_step(model, data)
            
            # Update viewer
            viewer.sync()
            
            # Control simulation speed
            time_until_next_step = step_start + 0.01 - time.time()
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
            
            step_count += 1

if __name__ == "__main__":
    main()
