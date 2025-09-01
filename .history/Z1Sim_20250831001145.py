import mujoco
import pinocchio as pin
import numpy as np
import time
import mujoco.viewer


class Z1Sim:
    def __init__(self, mjcf_path, urdf_path):
        # Load Mujoco model
        self.model = mujoco.MjModel.from_xml_path(mjcf_path)
        self.data = mujoco.MjData(self.model)
        # Load Pinocchio model
        self.pin_model = pin.buildModelFromUrdf(urdf_path)
        self.pin_data = self.pin_model.createData()
        # Simulation parameters
        self.dt = self.model.opt.timestep
        # Controller gains for position control
        self.kp = 50.0  # Proportional gain
        self.kd = 5.0  # Derivative gain
        # Number of joints (assuming 6-DOF arm)
        self.num_joints = 6

    def disable_dynamics(self):
        """Disable most dynamics effects for simplified simulation"""
        # Disable gravity
        self.model.opt.gravity[2] = 0.0
        # Disable joint damping
        self.model.dof_damping[:] = 0.0
        # Disable friction
        for i in range(self.model.ngeom):
            self.model.geom_friction[i][:] = 0.0
        # Disable armature (rotor inertia)
        self.model.dof_armature[:] = 0.0
        # Disable contact
        self.model.opt.disableflags = mujoco.mjtDisableBit.mjDSBL_CONTACT

    def reset(self, q=None, qdot=None):
        # Reset Mujoco state
        if q is not None:
            self.data.qpos[:] = q
        else:
            self.data.qpos[:] = 0
        if qdot is not None:
            self.data.qvel[:] = qdot
        else:
            self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)

    def step(self, ctrl):
        # Apply control and step simulation
        self.data.ctrl[:] = ctrl
        mujoco.mj_step(self.model, self.data)

    def get_state(self):
        # Return current state
        return np.copy(self.data.qpos), np.copy(self.data.qvel)

    def set_state(self, q, qdot):
        self.data.qpos[:] = q
        self.data.qvel[:] = qdot
        mujoco.mj_forward(self.model, self.data)

    def compute_pinocchio_dynamics(self, q, qdot, tau):
        # Use Pinocchio for forward dynamics
        pin.forwardDynamics(self.pin_model, self.pin_data, q, qdot, tau)
        return self.pin_data.ddq

    def compute_pinocchio_kinematics(self, q):
        # Use Pinocchio for forward kinematics
        pin.forwardKinematics(self.pin_model, self.pin_data, q)
        pin.updateFramePlacements(self.pin_model, self.pin_data)
        return self.pin_data.oMf

    def render(self):
        # Simple viewer
        mujoco.viewer.launch_passive(self.model, self.data)

    def send_torque(self, tau):
        # Send torque to actuators
        self.data.ctrl[: self.num_joints] = tau
        mujoco.mj_step(self.model, self.data)

    def position_control(self, target_positions):
        """
        Implements PD position control for the robot joints
        Args:
            target_positions: Array of target joint positions (radians)
        """
        # Get current state
        q, qdot = self.get_state()
        # Calculate position error
        pos_error = target_positions - q[: self.num_joints]
        # Calculate velocity error (target velocity is 0 for position control)
        vel_error = 0 - qdot[: self.num_joints]
        # PD control: τ = Kp * (qdes - q) + Kd * (qdot_des - qdot)
        torque = self.kp * pos_error + self.kd * vel_error
        # Apply torque command
        self.send_torque(torque)
        return pos_error

    def send_position(self, q_target):
        """
        Directly set the target positions for the actuators
        Args:
            q_target: Array of target joint positions (radians)
        """
        # In MuJoCo, when using position actuators, you set ctrl to desired positions
        self.data.ctrl[: self.num_joints] = q_target
        mujoco.mj_step(self.model, self.data)

    def solve_ik(
        self,
        target_position,
        target_orientation=None,
        initial_q=None,
        max_iter=100,
        eps=1e-4,
    ):
        """
        Solve inverse kinematics to find joint angles for a target end-effector position
        Args:
            target_position: 3D target position for the end-effector
            target_orientation: 3D target orientation (optional)
            initial_q: Initial joint configuration for IK solver (uses current if None)
            max_iter: Maximum number of iterations for the IK solver
            eps: Convergence threshold
        Returns:
            q_sol: Joint angles solution
            success: True if IK converged, False otherwise
        """
        # Get end effector frame ID (last operational frame)
        ee_frame_id = self.pin_model.getFrameId("link06")
        # Get initial configuration (current joint angles if not provided)
        if initial_q is None:
            q, _ = self.get_state()
            q = q[: self.num_joints].copy()
        else:
            q = initial_q.copy()
        # Initialize IK algorithm
        success = False
        for i in range(max_iter):
            # Forward kinematics to get current end effector position
            pin.forwardKinematics(self.pin_model, self.pin_data, q)
            pin.updateFramePlacements(self.pin_model, self.pin_data)
            # Get current end-effector position
            current_position = self.pin_data.oMf[ee_frame_id].translation
            # Compute position error
            pos_error = target_position - current_position
            # Check if we've converged
            if np.linalg.norm(pos_error) < eps:
                success = True
                break
            # Compute Jacobian
            J = pin.computeFrameJacobian(
                self.pin_model,
                self.pin_data,
                q,
                ee_frame_id,
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            )[
                :3, :
            ]  # Position only
            # Compute joint correction using damped least squares
            lambda_dls = 1e-6  # Damping factor
            J_pinv = J.T @ np.linalg.inv(J @ J.T + lambda_dls * np.eye(3))
            dq = J_pinv @ pos_error
            # Update joint angles
            q = q + dq * 0.5  # Step size of 0.5 for stability
            # Joint limits handling (if necessary)
            # q = np.clip(q, self.pin_model.lowerPositionLimit, self.pin_model.upperPositionLimit)
        return q, success


# Example usage:
if __name__ == "__main__":
    mjcf_path = "scene.xml"
    urdf_path = "z1.urdf"
    sim = Z1Sim(mjcf_path, urdf_path)
    sim.reset()
    # Create viewer
    viewer = mujoco.viewer.launch_passive(sim.model, sim.data)
    # Target positions (in radians) for each joint
    target_positions = np.array([0, 0.785, -0.261, -0.523, 0, 0])
    # Run direct position control loop
    start_time = time.time()
    while time.time() - start_time < 10:
        # Apply direct position control
        sim.send_position(target_positions)
        # Synchronize the viewer
        viewer.sync()
        # Get current state to monitor position
        q, _ = sim.get_state()
        error = target_positions - q[: sim.num_joints]
        # Print position error norm (optional)
        if int(time.time() * 10) % 10 == 0:  # print every ~0.1 seconds
            print(f"Position error: {np.linalg.norm(error):.4f}")
        time.sleep(0.01)  # Small delay for stability
    # Close viewer
    viewer.close()
    # Example usage for moving 10cm along x axis:
    # Create viewer
    viewer = mujoco.viewer.launch_passive(sim.model, sim.data)
    time.sleep(0.5)  # Wait for viewer to initialize
    # Get current end effector position
    q_init, _ = sim.get_state()
    q_init = q_init[: sim.num_joints]
    # Update Pinocchio data for current configuration
    pin.forwardKinematics(sim.pin_model, sim.pin_data, q_init)
    pin.updateFramePlacements(sim.pin_model, sim.pin_data)
    # Get end effector frame ID
    ee_frame_id = sim.pin_model.getFrameId("link06")
    # Get current end effector position
    current_ee_pos = sim.pin_data.oMf[ee_frame_id].translation
    print(f"Current end effector position: {current_ee_pos}")
    # Target position: 10cm along X axis from current position
    target_position = np.array([0.45, 0, 0.05])
    print(f"Target end effector position: {target_position}")
    # Solve IK
    q_solution, success = sim.solve_ik(target_position, initial_q=q_init)
    if success:
        print(f"IK solved successfully. Joint angles: {q_solution}")
        # Move to the solution position
        start_time = time.time()
        while time.time() - start_time < 3:  # 3 seconds of movement
            sim.send_position(q_solution)
            viewer.sync()
            time.sleep(0.01)
        # Verify final position
        q_final, _ = sim.get_state()
        pin.forwardKinematics(sim.pin_model, sim.pin_data, q_final[: sim.num_joints])
        pin.updateFramePlacements(sim.pin_model, sim.pin_data)
        final_ee_pos = sim.pin_data.oMf[ee_frame_id].translation
        print(f"Final end effector position: {final_ee_pos}")
        print(f"Error: {np.linalg.norm(final_ee_pos - target_position):.6f} meters")
    else:
        print("IK did not converge to a solution")
    # Keep viewer open for a few seconds
    time.sleep(3)
    viewer.close()
