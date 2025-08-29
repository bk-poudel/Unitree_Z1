#!/usr/bin/env python3
"""
Z1 Arm Controller with Pinocchio Dynamics and MuJoCo Visualization
Author: GitHub Copilot
Date: August 2025
This module provides a complete arm controller for the Unitree Z1 robot arm
that uses Pinocchio for dynamics computation and MuJoCo for visualization.
The controller implements impedance control for compliant robot behavior.
"""
import numpy as np
import mujoco
import mujoco.viewer

try:
    import pinocchio as pin

    PINOCCHIO_AVAILABLE = True
except ImportError:
    print("Warning: Pinocchio not available. Dynamics computations will be limited.")
    PINOCCHIO_AVAILABLE = False
from pathlib import Path
import time
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass


@dataclass
class ImpedanceParams:
    """Parameters for impedance control"""

    kp: np.ndarray  # Position stiffness (6x6 or 3x3)
    kd: np.ndarray  # Velocity damping (6x6 or 3x3)
    ki: np.ndarray  # Integral gain (6x6 or 3x3)
    max_force: float = 100.0  # Maximum force limit
    max_torque: float = 50.0  # Maximum torque limit


class Z1ArmController:
    """
    Unitree Z1 Arm Controller with Pinocchio dynamics and MuJoCo visualization.
    This class provides:
    - Forward/Inverse kinematics using Pinocchio
    - Dynamic model computation
    - Impedance control implementation
    - Real-time MuJoCo visualization
    - Joint and Cartesian space control
    """

    def __init__(
        self,
        urdf_path: Optional[str] = None,
        mujoco_xml_path: Optional[str] = None,
        dt: float = 0.001,
    ):
        """
        Initialize the Z1 arm controller.
        Args:
            urdf_path: Path to URDF file for Pinocchio
            mujoco_xml_path: Path to MuJoCo XML file
            dt: Control timestep
        """
        self.dt = dt
        self.current_time = 0.0
        # Set default paths if not provided
        if urdf_path is None:
            urdf_path = str(Path(__file__).parent / "z1.urdf")
        if mujoco_xml_path is None:
            mujoco_xml_path = str(Path(__file__).parent / "scene.xml")
        self.urdf_path = urdf_path
        self.mujoco_xml_path = mujoco_xml_path
        # Set number of joints first
        self.n_joints = 6
        # Initialize MuJoCo simulation first
        self._init_mujoco()
        # Initialize Pinocchio model if available
        self.pin_model = None
        self.pin_data = None
        self.ee_frame_id = None
        if PINOCCHIO_AVAILABLE:
            self._init_pinocchio()
        # Control variables
        self.q = np.zeros(self.n_joints)  # Joint positions
        self.dq = np.zeros(self.n_joints)  # Joint velocities
        self.tau = np.zeros(self.n_joints)  # Joint torques
        # Target variables
        self.q_des = np.zeros(self.n_joints)
        self.dq_des = np.zeros(self.n_joints)
        self.pose_des = np.eye(4)  # Desired end-effector pose
        self.twist_des = np.zeros(6)  # Desired end-effector twist
        # Impedance control variables
        self.integral_error = np.zeros(6)
        self.prev_error = np.zeros(6)
        # Default impedance parameters
        self.impedance_params = ImpedanceParams(
            kp=np.diag(
                [1000, 1000, 1000, 100, 100, 100]
            ),  # Position/orientation stiffness
            kd=np.diag([100, 100, 100, 10, 10, 10]),  # Damping
            ki=np.diag([10, 10, 10, 1, 1, 1]),  # Integral gain
        )
        print(f"Z1 Arm Controller initialized successfully!")
        print(f"Number of joints: {self.n_joints}")
        if PINOCCHIO_AVAILABLE and hasattr(self, "ee_frame_name"):
            print(f"End-effector frame: {self.ee_frame_name}")

    def _init_pinocchio(self):
        """Initialize Pinocchio model from URDF"""
        if not PINOCCHIO_AVAILABLE:
            print("Pinocchio not available, skipping dynamics initialization")
            return
        try:
            # Load the robot model
            self.pin_model = pin.buildModelFromUrdf(self.urdf_path)
            self.pin_data = self.pin_model.createData()
            # Get end-effector frame ID (assuming last link)
            self.ee_frame_name = "link06"
            if self.pin_model.existFrame(self.ee_frame_name):
                self.ee_frame_id = self.pin_model.getFrameId(self.ee_frame_name)
            else:
                # Fallback to last joint
                self.ee_frame_id = self.pin_model.nframes - 1
                self.ee_frame_name = self.pin_model.frames[self.ee_frame_id].name
            print(f"Pinocchio model loaded with {self.pin_model.nq} DOF")
        except Exception as e:
            print(f"Error loading Pinocchio model: {e}")
            self.pin_model = None
            self.pin_data = None

    def _init_mujoco(self):
        """Initialize MuJoCo simulation"""
        try:
            # Load MuJoCo model
            self.mj_model = mujoco.MjModel.from_xml_path(self.mujoco_xml_path)
            self.mj_data = mujoco.MjData(self.mj_model)
            # Set timestep
            self.mj_model.opt.timestep = self.dt
            # Get joint names and indices
            self.joint_names = []
            self.joint_ids = []
            for i in range(self.n_joints):
                joint_name = f"joint{i+1}"
                try:
                    joint_id = mujoco.mj_name2id(
                        self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
                    )
                    self.joint_names.append(joint_name)
                    self.joint_ids.append(joint_id)
                except:
                    print(f"Warning: Joint {joint_name} not found in MuJoCo model")
            print(f"MuJoCo model loaded with {self.mj_model.njnt} joints")
            print(f"Found joints: {self.joint_names}")
        except Exception as e:
            print(f"Error loading MuJoCo model: {e}")
            raise

    def update_state(self):
        """Update robot state from MuJoCo simulation"""
        # Get joint positions and velocities from MuJoCo
        for i, joint_id in enumerate(self.joint_ids):
            if joint_id >= 0 and joint_id < self.mj_model.nq:
                self.q[i] = self.mj_data.qpos[joint_id]
            if joint_id >= 0 and joint_id < self.mj_model.nv:
                self.dq[i] = self.mj_data.qvel[joint_id]
        # Update Pinocchio kinematics if available
        if self.pin_model is not None:
            pin.forwardKinematics(self.pin_model, self.pin_data, self.q, self.dq)
            pin.updateFramePlacements(self.pin_model, self.pin_data)

    def get_end_effector_pose(self) -> np.ndarray:
        """Get current end-effector pose as 4x4 homogeneous matrix"""
        if self.pin_model is not None and self.ee_frame_id is not None:
            return self.pin_data.oMf[self.ee_frame_id].homogeneous
        else:
            # Fallback: use MuJoCo's end-effector site or body
            # This is a simplified version - in practice you'd need to define
            # the end-effector location in your MuJoCo model
            ee_pos = np.array([0.5, 0.0, 0.3])  # Default position
            ee_rot = np.eye(3)  # Default orientation
            pose = np.eye(4)
            pose[:3, :3] = ee_rot
            pose[:3, 3] = ee_pos
            return pose

    def get_end_effector_velocity(self) -> np.ndarray:
        """Get current end-effector velocity (6D twist)"""
        if self.pin_model is not None:
            pin.computeFrameJacobian(
                self.pin_model, self.pin_data, self.q, self.ee_frame_id
            )
            J = self.pin_data.J
            return J @ self.dq
        else:
            return np.zeros(6)  # Fallback

    def get_jacobian(self) -> np.ndarray:
        """Get end-effector Jacobian matrix"""
        if self.pin_model is not None:
            pin.computeFrameJacobian(
                self.pin_model, self.pin_data, self.q, self.ee_frame_id
            )
            return self.pin_data.J.copy()
        else:
            # Return identity-like Jacobian as fallback
            return np.eye(6, self.n_joints)

    def get_mass_matrix(self) -> np.ndarray:
        """Get joint space mass matrix"""
        if self.pin_model is not None:
            pin.crba(self.pin_model, self.pin_data, self.q)
            return self.pin_data.M.copy()
        else:
            # Return identity matrix as fallback
            return np.eye(self.n_joints)

    def get_coriolis_forces(self) -> np.ndarray:
        """Get Coriolis and centrifugal forces"""
        if self.pin_model is not None:
            pin.computeCoriolisMatrix(self.pin_model, self.pin_data, self.q, self.dq)
            return self.pin_data.C @ self.dq
        else:
            return np.zeros(self.n_joints)  # Fallback

    def get_gravity_forces(self) -> np.ndarray:
        """Get gravity compensation torques"""
        if self.pin_model is not None:
            pin.computeGeneralizedGravity(self.pin_model, self.pin_data, self.q)
            return self.pin_data.g.copy()
        else:
            return np.zeros(self.n_joints)  # Fallback

    def inverse_kinematics(
        self,
        target_pose: np.ndarray,
        q_init: Optional[np.ndarray] = None,
        max_iter: int = 100,
        tol: float = 1e-6,
    ) -> Tuple[np.ndarray, bool]:
        """
        Solve inverse kinematics for target end-effector pose.
        Args:
            target_pose: 4x4 target pose matrix
            q_init: Initial joint configuration guess
            max_iter: Maximum iterations
            tol: Convergence tolerance
        Returns:
            Tuple of (joint_angles, success_flag)
        """
        if q_init is None:
            q_init = self.q.copy()
        if self.pin_model is None:
            print(
                "Warning: Pinocchio not available, returning current joint configuration"
            )
            return q_init, False
        q_sol = q_init.copy()
        target_placement = pin.SE3(target_pose[:3, :3], target_pose[:3, 3])
        for i in range(max_iter):
            pin.forwardKinematics(self.pin_model, self.pin_data, q_sol)
            pin.updateFramePlacements(self.pin_model, self.pin_data)
            # Compute error
            current_placement = self.pin_data.oMf[self.ee_frame_id]
            error = pin.log(current_placement.inverse() * target_placement)
            if np.linalg.norm(error.vector) < tol:
                return q_sol, True
            # Compute Jacobian and update
            pin.computeFrameJacobian(
                self.pin_model, self.pin_data, q_sol, self.ee_frame_id
            )
            J = self.pin_data.J
            # Damped least squares
            lambda_reg = 1e-6
            dq = np.linalg.solve(
                J.T @ J + lambda_reg * np.eye(self.n_joints), J.T @ error.vector
            )
            q_sol += dq * 0.1  # Step size
            # Clamp to joint limits (simplified)
            q_sol = np.clip(q_sol, -np.pi, np.pi)
        return q_sol, False

    def compute_impedance_control(
        self,
        target_pose: np.ndarray,
        target_twist: Optional[np.ndarray] = None,
        external_wrench: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute impedance control torques.
        Args:
            target_pose: Desired 4x4 end-effector pose
            target_twist: Desired 6D twist (optional)
            external_wrench: External wrench estimate (optional)
        Returns:
            Joint torques for impedance control
        """
        if target_twist is None:
            target_twist = np.zeros(6)
        if external_wrench is None:
            external_wrench = np.zeros(6)
        # Get current state
        current_pose = self.get_end_effector_pose()
        current_twist = self.get_end_effector_velocity()
        # Compute pose error (simplified for when Pinocchio is not available)
        if self.pin_model is not None:
            current_placement = pin.SE3(current_pose[:3, :3], current_pose[:3, 3])
            target_placement = pin.SE3(target_pose[:3, :3], target_pose[:3, 3])
            error_se3 = pin.log(current_placement.inverse() * target_placement)
            pose_error = error_se3.vector
        else:
            # Simplified pose error computation
            pos_error = target_pose[:3, 3] - current_pose[:3, 3]
            # Simplified orientation error (axis-angle approximation)
            rot_error = np.array([0.0, 0.0, 0.0])  # Placeholder
            pose_error = np.concatenate([pos_error, rot_error])
        # Compute velocity error
        twist_error = target_twist - current_twist
        # Integral error update
        self.integral_error += pose_error * self.dt
        # Compute desired Cartesian force/torque
        desired_wrench = (
            self.impedance_params.kp @ pose_error
            + self.impedance_params.kd @ twist_error
            + self.impedance_params.ki @ self.integral_error
        )
        # Add external wrench compensation
        desired_wrench += external_wrench
        # Limit forces and torques
        desired_wrench[:3] = np.clip(
            desired_wrench[:3],
            -self.impedance_params.max_force,
            self.impedance_params.max_force,
        )
        desired_wrench[3:] = np.clip(
            desired_wrench[3:],
            -self.impedance_params.max_torque,
            self.impedance_params.max_torque,
        )
        # Transform to joint space
        J = self.get_jacobian()
        joint_torques = J.T @ desired_wrench
        # Add gravity compensation
        gravity_comp = self.get_gravity_forces()
        return joint_torques + gravity_comp

    def set_impedance_params(
        self, kp: np.ndarray, kd: np.ndarray, ki: Optional[np.ndarray] = None
    ):
        """Set impedance control parameters"""
        if ki is None:
            ki = np.zeros_like(kp)
        self.impedance_params.kp = kp.copy()
        self.impedance_params.kd = kd.copy()
        self.impedance_params.ki = ki.copy()
        # Reset integral error
        self.integral_error = np.zeros(6)

    def set_target_pose(self, pose: np.ndarray, twist: Optional[np.ndarray] = None):
        """Set target end-effector pose and twist"""
        self.pose_des = pose.copy()
        if twist is not None:
            self.twist_des = twist.copy()
        else:
            self.twist_des = np.zeros(6)

    def step_simulation(self, control_torques: Optional[np.ndarray] = None):
        """Step the MuJoCo simulation forward"""
        if control_torques is not None:
            # Apply control torques
            for i, joint_id in enumerate(self.joint_ids):
                if (
                    i < len(control_torques)
                    and joint_id >= 0
                    and joint_id < self.mj_model.nu
                ):
                    self.mj_data.ctrl[joint_id] = control_torques[i]
        # Step simulation
        mujoco.mj_step(self.mj_model, self.mj_data)
        self.current_time += self.dt
        # Update state
        self.update_state()

    def run_impedance_control_loop(
        self, target_pose: np.ndarray, duration: float = 10.0, visualize: bool = True
    ):
        """
        Run impedance control loop for specified duration.
        Args:
            target_pose: Target end-effector pose
            duration: Control duration in seconds
            visualize: Whether to show MuJoCo viewer
        """
        self.set_target_pose(target_pose)
        viewer = None
        if visualize:
            viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)
        start_time = time.time()
        try:
            while (time.time() - start_time) < duration:
                # Compute control torques
                tau_control = self.compute_impedance_control(
                    self.pose_des, self.twist_des
                )
                # Step simulation
                self.step_simulation(tau_control)
                # Update viewer
                if viewer is not None:
                    viewer.sync()
                # Sleep to maintain real-time
                time.sleep(max(0, self.dt - (time.time() % self.dt)))
        except KeyboardInterrupt:
            print("Control loop interrupted by user")
        finally:
            if viewer is not None:
                viewer.close()

    def get_current_state(self) -> Dict[str, Any]:
        """Get current robot state"""
        return {
            "joint_positions": self.q.copy(),
            "joint_velocities": self.dq.copy(),
            "end_effector_pose": self.get_end_effector_pose(),
            "end_effector_velocity": self.get_end_effector_velocity(),
            "time": self.current_time,
        }

    def reset_to_home(self):
        """Reset robot to home configuration"""
        home_config = np.zeros(self.n_joints)
        # Set MuJoCo state
        for i, joint_id in enumerate(self.joint_ids):
            if joint_id >= 0 and joint_id < self.mj_model.nq:
                self.mj_data.qpos[joint_id] = home_config[i]
            if joint_id >= 0 and joint_id < self.mj_model.nv:
                self.mj_data.qvel[joint_id] = 0.0
        # Forward simulation step
        mujoco.mj_forward(self.mj_model, self.mj_data)
        self.update_state()
        # Reset control variables
        self.integral_error = np.zeros(6)
        self.current_time = 0.0

    def run_joint_position_control(
        self,
        target_positions: np.ndarray,
        duration: float = 5.0,
        visualize: bool = True,
    ):
        """
        Run simple joint position control (PD control in joint space).
        Args:
            target_positions: Target joint positions (rad)
            duration: Control duration in seconds
            visualize: Whether to show MuJoCo viewer
        """
        # Simple PD gains
        kp_joint = 1000.0
        kd_joint = 100.0
        viewer = None
        if visualize:
            viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)
        start_time = time.time()
        try:
            while (time.time() - start_time) < duration:
                # Simple PD control in joint space
                position_error = target_positions - self.q
                velocity_error = -self.dq  # Desired velocity is zero
                tau_control = kp_joint * position_error + kd_joint * velocity_error
                # Add gravity compensation if available
                tau_control += self.get_gravity_forces()
                # Step simulation
                self.step_simulation(tau_control)
                # Update viewer
                if viewer is not None:
                    viewer.sync()
                # Sleep to maintain real-time
                time.sleep(max(0, self.dt))
        except KeyboardInterrupt:
            print("Control loop interrupted by user")
        finally:
            if viewer is not None:
                viewer.close()


def main():
    """Example usage of the Z1ArmController"""
    # Initialize controller
    try:
        controller = Z1ArmController()
    except Exception as e:
        print(f"Failed to initialize controller: {e}")
        print("Make sure z1.xml exists in the current directory")
        return
    # Set impedance parameters (softer for demonstration)
    kp = np.diag([500, 500, 500, 50, 50, 50])  # Position/orientation stiffness
    kd = np.diag([50, 50, 50, 5, 5, 5])  # Damping
    controller.set_impedance_params(kp, kd)
    # Reset to home position
    controller.reset_to_home()
    print("Current end-effector pose:")
    print(controller.get_end_effector_pose())
    # Example 1: Joint space control
    print("\n=== Running joint position control ===")
    target_joints = np.array([0.5, 0.3, -0.5, 0.2, 0.1, 0.0])
    print("Moving to joint configuration:", target_joints)
    controller.run_joint_position_control(target_joints, duration=10.0, visualize=True)
    # Example 2: Cartesian impedance control
    print("\n=== Running Cartesian impedance control ===")
    # Define target pose (move 10cm forward in X)
    target_pose = controller.get_end_effector_pose().copy()
    target_pose[0, 3] += 0.1  # Move 10cm in X direction
    print(f"Moving to target pose...")
    print("Press Ctrl+C to stop")
    # Run impedance control
    controller.run_impedance_control_loop(
        target_pose=target_pose, duration=15.0, visualize=True  # 15 seconds
    )


if __name__ == "__main__":
    main()
