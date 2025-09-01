import mujoco
import numpy as np
import mujoco.viewer as viewer
import time


def analyze_model(model):
    """Analyze the model configuration"""
    print("=" * 50)
    print("MODEL ANALYSIS")
    print("=" * 50)
    print(f"Number of DOFs: {model.nv}")
    print(f"Number of joints: {model.njnt}")
    print(f"Number of actuators: {model.nu}")
    print("\nJOINT INFORMATION:")
    for i in range(model.njnt):
        joint_name = model.joint(i).name
        joint_type = model.jnt_type[i]
        joint_limited = model.jnt_limited[i]
        joint_range = model.jnt_range[i]
        joint_axis = model.jnt_axis[i]
        type_names = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}
        print(f"  Joint {i}: {joint_name}")
        print(f"    Type: {type_names.get(joint_type, 'unknown')}")
        print(f"    Axis: {joint_axis}")
        print(f"    Limited: {joint_limited}")
        print(f"    Range: [{joint_range[0]:.3f}, {joint_range[1]:.3f}]")
    print("\nACTUATOR INFORMATION:")
    for i in range(model.nu):
        actuator_name = model.actuator(i).name
        ctrl_range = model.actuator_ctrlrange[i]
        force_range = model.actuator_forcerange[i]
        gear = model.actuator_gear[i]
        print(f"  Actuator {i}: {actuator_name}")
        print(f"    Control range: [{ctrl_range[0]:.1f}, {ctrl_range[1]:.1f}]")
        print(f"    Force range: [{force_range[0]:.1f}, {force_range[1]:.1f}]")
        print(f"    Gear: {gear}")


def test_individual_joints(model, data, viewer_handle):
    """Test each joint individually"""
    print("\n" + "=" * 50)
    print("TESTING INDIVIDUAL JOINTS")
    print("=" * 50)
    # Test each actuator
    for joint_idx in range(min(model.nu, 6)):  # Test first 6 actuators
        print(f"\nTesting joint {joint_idx} ({model.actuator(joint_idx).name})...")
        # Reset to home position
        home_pos = np.array([0, 0.785, -0.261, -0.523, 0, 0])
        data.qpos[:6] = home_pos
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        # Apply torque to this joint only
        test_torques = [0.5, 1.0, 2.0, 5.0, 10.0]  # Increasing torque levels
        for torque in test_torques:
            print(f"  Trying torque {torque} N⋅m...")
            # Reset position
            data.qpos[:6] = home_pos
            data.qvel[:] = 0
            data.ctrl[:] = 0
            initial_pos = data.qpos[joint_idx]
            # Apply torque for 1000 steps (1 second)
            for step in range(1000):
                data.ctrl[joint_idx] = torque
                mujoco.mj_step(model, data)
                if step % 200 == 0:  # Update viewer every 200ms
                    viewer_handle.sync()
            final_pos = data.qpos[joint_idx]
            final_vel = data.qvel[joint_idx]
            movement = abs(final_pos - initial_pos)
            print(f"    Initial pos: {initial_pos:.4f}, Final pos: {final_pos:.4f}")
            print(f"    Movement: {movement:.4f} rad ({np.degrees(movement):.2f}°)")
            print(f"    Final velocity: {final_vel:.4f} rad/s")
            if movement > 0.01:  # Significant movement detected
                print(f"    ✓ Joint {joint_idx} MOVED with {torque} N⋅m!")
                break
            else:
                print(f"    ✗ Joint {joint_idx} did not move significantly")
        time.sleep(0.5)  # Pause between joint tests


def test_gravity_effects(model, data):
    """Test if gravity is affecting the robot"""
    print("\n" + "=" * 50)
    print("TESTING GRAVITY EFFECTS")
    print("=" * 50)
    # Check gravity setting
    gravity = model.opt.gravity
    print(f"Gravity vector: {gravity}")
    # Set robot to home position and check gravitational torques
    home_pos = np.array([0, 0.785, -0.261, -0.523, 0, 0])
    data.qpos[:6] = home_pos
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    print("Gravitational torques at home position:")
    for i in range(min(6, model.nu)):
        gravity_torque = data.qfrc_bias[i] if i < len(data.qfrc_bias) else 0
        print(f"  Joint {i}: {gravity_torque:.4f} N⋅m")


def main():
    # Load model
    print("Loading model...")
    try:
        model = mujoco.MjModel.from_xml_path("./scene.xml")
        data = mujoco.MjData(model)
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return
    # Analyze model
    analyze_model(model)
    # Test gravity effects
    test_gravity_effects(model, data)
    # Launch viewer
    print("\nLaunching viewer...")
    viewer_handle = viewer.launch_passive(model, data)
    # Wait for viewer to initialize
    time.sleep(1)
    # Test individual joints
    try:
        test_individual_joints(model, data, viewer_handle)
    except KeyboardInterrupt:
        print("\nTesting interrupted by user")
    print("\n" + "=" * 50)
    print("FINAL CONTINUOUS ROTATION TEST")
    print("=" * 50)
    print("Testing continuous rotation on joint 0 with 3.0 N⋅m...")
    print("Press Ctrl+C to stop")
    # Reset to home position
    home_pos = np.array([0, 0.785, -0.261, -0.523, 0, 0])
    data.qpos[:6] = home_pos
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    step_count = 0
    start_time = time.time()
    try:
        while viewer_handle.is_running():
            # Apply constant torque to joint 0
            gravity = data.qfrc_bias[0] if len(data.qfrc_bias) > 0 else 0
            data.ctrl[:] = [0.6, 0, 0, 0, 0, 0] + gravity  # Strong torque
            mujoco.mj_step(model, data)
            viewer_handle.sync()
            # Print status every 2 seconds
            if step_count % 2000 == 0:
                elapsed = time.time() - start_time
                pos = data.qpos[0]
                vel = data.qvel[0]
                print(
                    f"t={elapsed:.1f}s: pos={pos:.3f}rad ({np.degrees(pos):.1f}°), vel={vel:.3f}rad/s"
                )
            step_count += 1
            time.sleep(0.001)
    except KeyboardInterrupt:
        print("\nFinal test stopped by user")
    finally:
        viewer_handle.close()
    print("Analysis complete!")


if __name__ == "__main__":
    main()
