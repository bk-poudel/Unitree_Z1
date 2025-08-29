#!/usr/bin/env python3
import numpy as np
import time
from Z1Sim import Z1Sim


def main():
    # Initialize the Z1 simulator
    print("Initializing Z1 Simulator...")
    z1 = Z1Sim(
        interface_type="torque",  # Use torque control
        render=True,  # Enable visualization
        dt=0.001,  # 1ms timestep
    )
    try:
        # Example 1: Basic robot state monitoring
        print("\n=== Example 1: Basic State Monitoring ===")
        for i in range(100):
            q, v = z1.get_state()
            print(
                f"Step {i}: Joint positions: {[f'{x:.3f}' for x in q[:3]]}, velocities: {[f'{x:.3f}' for x in v[:3]]}"
            )
            time.sleep(0.01)
        # Example 2: Reset robot to home position
        print("\n=== Example 2: Reset to Home Position ===")
        z1.reset()
        q, v = z1.get_state()
        print(f"After reset - Positions: {q}")
        # Example 3: Forward kinematics
        print("\n=== Example 3: Forward Kinematics ===")
        q, _ = z1.get_state()
        ee_pose = z1.get_pose(q)
        print(f"End-effector pose:\n{ee_pose}")
        # Example 4: Jacobian computation
        print("\n=== Example 4: Jacobian Computation ===")
        J = z1.get_jacobian(q)
        print(f"Jacobian shape: {J.shape}")
        print(f"Jacobian:\n{J}")
        # Example 5: Gravity compensation
        print("\n=== Example 5: Gravity Compensation ===")
        for i in range(200):
            q, v = z1.get_state()
            g = z1.get_gravity(q)
            # Apply gravity compensation torque
            z1.send_joint_torque(g)
            time.sleep(0.01)
        # Example 6: Simple position control
        print("\n=== Example 6: Simple Position Control ===")
        target_q = np.array([0.5, 0.8, -1.2, 0.3, 0.2, 0.1])
        kp = 50.0  # Proportional gain
        kd = 5.0  # Derivative gain
        for i in range(500):
            q, v = z1.get_state()
            # PD control
            tau_pd = kp * (target_q - q) - kd * v
            # Add gravity compensation
            g = z1.get_gravity(q)
            tau_total = tau_pd + g
            z1.send_joint_torque(tau_total)
            if i % 50 == 0:
                error = np.linalg.norm(target_q - q)
                print(f"Step {i}: Position error: {error:.4f}")
            time.sleep(0.01)
        # Example 7: Cartesian space control
        print("\n=== Example 7: Cartesian Space Control ===")
        # Define target end-effector position
        target_pos = np.array([0.3, 0.2, 0.4])
        kp_cart = 100.0
        kd_cart = 10.0
        for i in range(300):
            q, v = z1.get_state()
            # Get current end-effector pose
            ee_pose = z1.get_pose(q)
            current_pos = ee_pose[:3, 3]
            # Position error in Cartesian space
            pos_error = target_pos - current_pos
            # Get Jacobian
            J = z1.get_jacobian(q)
            J_pos = J[:3, :]  # Position part of Jacobian
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
            if i % 30 == 0:
                cart_error = np.linalg.norm(pos_error)
                print(f"Step {i}: Cartesian error: {cart_error:.4f}")
            time.sleep(0.01)
        # Example 8: Trajectory following
        print("\n=== Example 8: Trajectory Following ===")
        # Create a circular trajectory in Cartesian space
        center = np.array([0.3, 0.0, 0.4])
        radius = 0.1
        num_points = 200
        for i in range(num_points):
            angle = 2 * np.pi * i / num_points
            target_pos = center + radius * np.array([np.cos(angle), np.sin(angle), 0])
            q, v = z1.get_state()
            ee_pose = z1.get_pose(q)
            current_pos = ee_pose[:3, 3]
            pos_error = target_pos - current_pos
            J = z1.get_jacobian(q)
            J_pos = J[:3, :]
            tau_cart = J_pos.T @ (kp_cart * pos_error)
            g = z1.get_gravity(q)
            tau_total = tau_cart + g
            z1.send_joint_torque(tau_total)
            if i % 20 == 0:
                print(
                    f"Trajectory point {i}: Target {target_pos}, Current {current_pos}"
                )
            time.sleep(0.02)
        # Example 9: Force/torque sensing
        print("\n=== Example 9: Force/Torque Sensing ===")
        for i in range(50):
            force, torque = z1.get_ee_force_torque()
            print(f"EE Force: {force}, Torque: {torque}")
            # Apply small torque
            q, v = z1.get_state()
            g = z1.get_gravity(q)
            z1.send_joint_torque(g)
            time.sleep(0.02)
        # Example 10: Using ROS integration
        print("\n=== Example 10: ROS Integration ===")
        # Send torque commands via ROS
        for i in range(50):
            q, v = z1.get_state()
            g = z1.get_gravity(q)
            # Send torque to ROS topic
            z1.send_torque_to_ros_node(g)
            # Get latest joint states from ROS
            joint_states = z1.get_latest_joint_states()
            if joint_states:
                print(f"ROS Joint States: {joint_states['positions'][:3]}")
            time.sleep(0.02)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        # Clean shutdown
        print("Closing simulator...")
        z1.close()


if __name__ == "__main__":
    main()
