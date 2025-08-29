#!/usr/bin/env python3
import numpy as np
import time
from Z1Sim import Z1Sim


def main():
    print("Initializing Z1 Simulator...")
    z1 = Z1Sim(
        interface_type="torque",
        render=True,
        dt=0.001,
    )
    try:
        print("\n=== Moving 10cm in X direction ===")
        # Get initial position
        q_initial, _ = z1.get_state()
        print("Using Cartesian space control")
        # Get current end-effector position
        ee_pose = z1.get_pose(q_initial)
        current_pos = ee_pose[:3, 3]
        # Target: move 10cm (0.1m) in Z direction
        target_pos = current_pos.copy()
        target_pos[2] -= 0.1  # 10cm in Z
        print(f"Initial position: {current_pos}")
        print(f"Target position: {target_pos}")
        kp_cart = 100.0
        kd_cart = 10.0
        for i in range(1000):
            q, v = z1.get_state()
            # Get current end-effector pose
            ee_pose = z1.get_pose(q)
            current_pos = ee_pose[:3, 3]
            # Position error
            pos_error = target_pos - current_pos
            error_magnitude = np.linalg.norm(pos_error)
            # Check if we've reached the target
            if error_magnitude < 0.005:  # 5mm tolerance
                print("Target reached!")
                break
            # Get Jacobian
            J = z1.get_jacobian(q)
            J_pos = J[:3, :]  # Position part only
            # Cartesian PD control
            desired_vel = kp_cart * pos_error
            current_vel = J_pos @ v
            vel_error = desired_vel - current_vel
            # Map to joint space
            tau_cart = J_pos.T @ (kp_cart * pos_error + kd_cart * vel_error)
            # Add gravity compensation
            g = z1.get_gravity(q)
            tau_total = tau_cart + g
            z1.send_joint_torque(tau_total)
            # Print progress
            if i % 50 == 0:
                print(f"Step {i}: Distance to target: {error_magnitude*1000:.1f}mm")
                print(f"Current Z: {current_pos[2]:.3f}, Target Z: {target_pos[2]:.3f}")
            time.sleep(0.01)
        print("Movement completed!")
        # Hold position for a moment
        print("Holding position...")
        for i in range(100):
            q, v = z1.get_state()
            if z1.pin_model:
                g = z1.get_gravity(q)
                z1.send_joint_torque(g)
            else:
                z1.send_joint_torque(np.zeros(6))
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        print("Closing simulator...")
        z1.close()


if __name__ == "__main__":
    main()
