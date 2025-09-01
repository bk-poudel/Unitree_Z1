import mujoco
import numpy as np
import mujoco.viewer as viewer
import time


def analyze_model(model):
    """Analyze joints and actuators in the model"""
    print("=" * 50)
    print("MODEL ANALYSIS")
    print("=" * 50)
    print(f"Degrees of Freedom (DOFs): {model.nv}")
    print(f"Number of joints: {model.njnt}")
    print(f"Number of actuators: {model.nu}")
    type_names = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}
    print("\nJOINT INFORMATION:")
    for i in range(model.njnt):
        joint = model.joint(i)
        joint_type = model.jnt_type[i]
        joint_limited = model.jnt_limited[i]
        joint_range = model.jnt_range[i]
        joint_axis = model.jnt_axis[i]
        print(f"  Joint {i}: {joint.name}")
        print(f"    Type: {type_names.get(joint_type, 'unknown')}")
        print(f"    Axis: {joint_axis}")
        print(f"    Limited: {joint_limited}")
        print(f"    Range: [{joint_range[0]:.3f}, {joint_range[1]:.3f}]")
    print("\nACTUATOR INFORMATION:")
    for i in range(model.nu):
        actuator = model.actuator(i)
        ctrl_range = model.actuator_ctrlrange[i]
        force_range = model.actuator_forcerange[i]
        gear = model.actuator_gear[i]
        print(f"  Actuator {i}: {actuator.name}")
        print(f"    Control range: [{ctrl_range[0]:.1f}, {ctrl_range[1]:.1f}]")
        print(f"    Force range: [{force_range[0]:.1f}, {force_range[1]:.1f}]")
        print(f"    Gear: {gear}")


def test_individual_joints(model, data, viewer_handle, steps_per_torque=1000):
    """Test each joint with increasing torque"""
    print("\n" + "=" * 50)
    print("TESTING INDIVIDUAL JOINTS")
    print("=" * 50)
    nq = model.nq
    home_pos = np.zeros(nq)  # Default home position
    for joint_idx in range(min(model.nu, 6)):
        actuator = model.actuator(joint_idx)
        print(f"\nTesting joint {joint_idx} ({actuator.name})...")
        # Torque levels (safe range)
        min_torque, max_torque = model.actuator_forcerange[joint_idx]
        test_torques = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        test_torques = [min(max_torque, t) for t in test_torques]
        for torque in test_torques:
            # Reset robot state
            data.qpos[:] = home_pos
            data.qvel[:] = 0
            data.ctrl[:] = 0
            mujoco.mj_forward(model, data)
            initial_pos = data.qpos[joint_idx]
            # Apply torque
            for step in range(steps_per_torque):
                data.ctrl[:] = 0
                data.ctrl[joint_idx] = torque
                mujoco.mj_step(model, data)
                if step % 200 == 0:
                    viewer_handle.sync()
            final_pos = data.qpos[joint_idx]
            movement = abs(final_pos - initial_pos)
            print(
                f"  Torque {torque} N⋅m -> Movement: {movement:.4f} rad ({np.degrees(movement):.2f}°)"
            )
            if movement > 0.01:
                print(f"    ✓ Joint {joint_idx} MOVED with {torque} N⋅m")
                break
            else:
                print(f"    ✗ Joint {joint_idx} did not move significantly")


def test_gravity_effects(model, data):
    """Check gravitational torques"""
    print("\n" + "=" * 50)
    print("TESTING GRAVITY EFFECTS")
    print("=" * 50)
    print(f"Gravity vector: {model.opt.gravity}")
    data.qpos[:] = np.zeros(model.nq)
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    print("Gravitational torques at home position:")
    for i in range(model.nu):
        gravity_torque = data.qfrc_bias[i]
        print(f"  Joint {i}: {gravity_torque:.4f} N⋅m")


def continuous_torque_test(model, data, viewer_handle, joint_idx=0, torque=1.0):
    """Apply continuous torque to a joint"""
    print("\n" + "=" * 50)
    print("CONTINUOUS TORQUE TEST")
    print("=" * 50)
    print(f"Applying {torque} N⋅m on joint {joint_idx}. Press Ctrl+C to stop.")
    home_pos = np.zeros(model.nq)
    data.qpos[:] = home_pos
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    step_count = 0
    start_time = time.time()
    try:
        while viewer_handle.is_running():
            data.ctrl[:] = 0
            data.ctrl[joint_idx] = torque
            mujoco.mj_step(model, data)
            viewer_handle.sync()
            if step_count % 2000 == 0:
                elapsed = time.time() - start_time
                pos = data.qpos[joint_idx]
                vel = data.qvel[joint_idx]
                print(
                    f"t={elapsed:.1f}s: pos={pos:.3f} rad ({np.degrees(pos):.1f}°), vel={vel:.3f} rad/s"
                )
            step_count += 1
            time.sleep(0.001)
    except KeyboardInterrupt:
        print("\nTest stopped by user")


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
    # Analyze
    analyze_model(model)
    # Gravity test
    test_gravity_effects(model, data)
    # Launch viewer
    print("\nLaunching viewer...")
    viewer_handle = viewer.launch_passive(model, data)
    time.sleep(1)
    # Individual joint tests
    test_individual_joints(model, data, viewer_handle)
    # Continuous torque test
    continuous_torque_test(model, data, viewer_handle, joint_idx=0, torque=3.0)
    viewer_handle.close()
    print("Analysis complete!")


if __name__ == "__main__":
    main()
