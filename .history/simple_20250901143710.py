import mujoco
import pinocchio as pin
import numpy as np
import mujoco.viewer as viewer
import time

model = mujoco.MjModel.from_xml_path("./scene.xml")
data = mujoco.MjData(model)
dt = 0.001
# Set home position
home_pos = np.array([0, 0.785, -0.261, -0.523, 0, 0])
data.qpos[:6] = home_pos
print("Joint names:")
for i in range(model.njnt):
    print(f"  {i}: {model.joint(i).name}")
print("\nActuator names and torque ranges:")
for i in range(model.nu):
    ctrl_range = model.actuator_ctrlrange[i]
    force_range = model.actuator_forcerange[i]
    print(f"  {i}: {model.actuator(i).name}")
    print(f"     Control range: [{ctrl_range[0]:.1f}, {ctrl_range[1]:.1f}] N⋅m")
    print(f"     Force range: [{force_range[0]:.1f}, {force_range[1]:.1f}] N⋅m")
print("\nBody names:")
for i in range(model.nbody):
    print(f"  {i}: {model.body(i).name}")
# Initialize simulation - important to call mj_forward first
mujoco.mj_forward(model, data)
# Launch viewer
viewer_handle = viewer.launch_passive(model, data)
# Control parameters
applied_torque = 0.3  # N⋅m - start small
step_count = 0
start_time = time.time()
print(f"\nStarting simulation with {applied_torque} N⋅m torque on joint 1...")
print("Press Ctrl+C to stop\n")
try:
    while viewer_handle.is_running():
        # Apply torque only to first joint, others stay at zero
        data.ctrl[:] = 0  # Reset all controls
        data.ctrl[0] = applied_torque  # Apply torque to joint 1
        # Step the simulation
        mujoco.mj_step(model, data)
        # Sync with viewer
        viewer_handle.sync()
        # Print debug info every 2000 steps (every 2 seconds)
        if step_count % 2000 == 0:
            elapsed_time = time.time() - start_time
            print(f"Time: {elapsed_time:.1f}s | Step: {step_count}")
            print(
                f"  Joint 1 position: {data.qpos[0]:.3f} rad ({np.degrees(data.qpos[0]):.1f}°)"
            )
            print(f"  Joint 1 velocity: {data.qvel[0]:.3f} rad/s")
            print(f"  Applied torque: {data.ctrl[0]:.3f} N⋅m")
            print(f"  All joint positions: {data.qpos[:6]}")
            print(f"  All joint velocities: {data.qvel[:6]}")
            print()
        # Optional: Increase torque after 5 seconds if no movement
        if step_count == 5000 and abs(data.qvel[0]) < 0.01:
            applied_torque = 1.0
            print(f"No movement detected, increasing torque to {applied_torque} N⋅m")
        # Optional: Increase torque after 10 seconds if still no movement
        if step_count == 10000 and abs(data.qvel[0]) < 0.01:
            applied_torque = 3.0
            print(f"Still no movement, increasing torque to {applied_torque} N⋅m")
        step_count += 1
        time.sleep(dt)
except KeyboardInterrupt:
    print("\nSimulation stopped by user")
    print(
        f"Final joint 1 position: {data.qpos[0]:.3f} rad ({np.degrees(data.qpos[0]):.1f}°)"
    )
    print(f"Final joint 1 velocity: {data.qvel[0]:.3f} rad/s")
finally:
    viewer_handle.close()
