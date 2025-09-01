import mujoco
import pinocchio as pin
import numpy as np
import mujoco.viewer as viewer
import time

# Load model and create data
model = mujoco.MjModel.from_xml_path("./scene.xml")
data = mujoco.MjData(model)
dt = 0.001
# Set home position
home_pos = np.array([0, 0.785, -0.261, -0.523, 0, 0])
data.qpos[:6] = home_pos
# Print model information
print("Joint names:")
for i in range(model.njnt):
    print(f"  {i}: {model.joint(i).name}")
print("\nActuator names:")
for i in range(model.nu):
    print(f"  {i}: {model.actuator(i).name}")
print("\nActuator control ranges:")
for i in range(model.nu):
    ctrl_range = model.actuator_ctrlrange[i]
    print(f"  {model.actuator(i).name}: [{ctrl_range[0]:.3f}, {ctrl_range[1]:.3f}]")
print("\nBody names:")
for i in range(model.nbody):
    print(f"  {i}: {model.body(i).name}")
# Check actuator types
print("\nActuator types:")
for i in range(model.nu):
    actuator_type = model.actuator_dyntype[i]
    actuator_name = model.actuator(i).name
    type_names = {0: "none", 1: "integrator", 2: "filter", 3: "muscle", 4: "user"}
    print(f"  {actuator_name}: {type_names.get(actuator_type, 'unknown')}")
# Initialize simulation
mujoco.mj_forward(model, data)  # Compute forward dynamics once
# Launch viewer
viewer_handle = viewer.launch_passive(model, data)
# Control parameters
torque_command = 0.3  # N⋅m
step_count = 0
try:
    while viewer_handle.is_running():
        # Apply torque to first joint only
        data.ctrl[:] = 0  # Reset all controls
        data.ctrl[0] = torque_command  # Apply torque to first actuator
        # Step simulation
        mujoco.mj_step(model, data)
        # Sync viewer
        viewer_handle.sync()
        # Print debug info every 1000 steps
        if step_count % 1000 == 0:
            print(f"Step {step_count}:")
            print(f"  Joint positions: {data.qpos[:6]}")
            print(f"  Joint velocities: {data.qvel[:6]}")
            print(f"  Applied torques: {data.ctrl[:6]}")
            print(f"  Actual torques: {data.qfrc_applied[:6]}")
        step_count += 1
        time.sleep(dt)
except KeyboardInterrupt:
    print("Simulation stopped by user")
finally:
    viewer_handle.close()
