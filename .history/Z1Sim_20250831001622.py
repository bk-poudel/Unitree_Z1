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
        orientation_weight=1.0,
    ):
        """
        Solve inverse kinematics to find joint angles for a target end-effector position
        while maintaining orientation
        Args:
            target_position: 3D target position for the end-effector
            target_orientation: Target orientation as rotation matrix (if None, keeps initial orientation)
            initial_q: Initial joint configuration for IK solver (uses current if None)
            max_iter: Maximum number of iterations for the IK solver
            eps: Convergence threshold
            orientation_weight: Weight of orientation vs position task (higher = prioritize orientation)
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
        # Calculate initial orientation if target not provided
        pin.forwardKinematics(self.pin_model, self.pin_data, q)
        pin.updateFramePlacements(self.pin_model, self.pin_data)
        initial_orientation = self.pin_data.oMf[ee_frame_id].rotation.copy()
        if target_orientation is None:
            target_orientation = initial_orientation
        # Initialize IK algorithm
        success = False
        for i in range(max_iter):
            # Forward kinematics to get current end effector pose
            pin.forwardKinematics(self.pin_model, self.pin_data, q)
            pin.updateFramePlacements(self.pin_model, self.pin_data)
            # Get current end-effector position and orientation
            current_position = self.pin_data.oMf[ee_frame_id].translation
            current_orientation = self.pin_data.oMf[ee_frame_id].rotation
            # Compute position error
            pos_error = target_position - current_position
            # Compute orientation error (convert to rotation vector)
            orientation_error = pin.log3(target_orientation @ current_orientation.T)
            # Combine errors into task error vector (position and orientation)
            task_error = np.concatenate(
                [pos_error, orientation_weight * orientation_error]
            )
            # Check if we've converged (for position only, can be modified)
            if (
                np.linalg.norm(pos_error) < eps
                and np.linalg.norm(orientation_error) < eps
            ):
                success = True
                break
            # Compute Jacobian (position and orientation)
            J = pin.computeFrameJacobian(
                self.pin_model,
                self.pin_data,
                q,
                ee_frame_id,
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            )
            J_pos = J[:3, :]  # Position Jacobian
            J_orient = J[3:6, :]  # Orientation Jacobian
            # Combine into task Jacobian
            task_J = np.vstack([J_pos, orientation_weight * J_orient])
            # Compute joint correction using damped least squares
            lambda_dls = 1e-6  # Damping factor
            J_pinv = task_J.T @ np.linalg.inv(
                task_J @ task_J.T + lambda_dls * np.eye(6)
            )
            dq = J_pinv @ task_error
            # Update joint angles
            q = q + dq * 0.3  # Reduced step size for stability with orientation
            # Joint limits handling (if necessary)
            # q = np.clip(q, self.pin_model.lowerPositionLimit, self.pin_model.upperPositionLimit)
        return q, success

    def interpolate_trajectory(
        self,
        start_position,
        target_position,
        orientation=None,
        num_steps=20,
        initial_q=None,
    ):
        """
        Interpolate a trajectory between start and target positions while maintaining orientation.
        Solves IK for each step of the trajectory.
        Args:
            start_position: 3D starting position for the end-effector
            target_position: 3D target position for the end-effector
            orientation: Target orientation to maintain (if None, uses current orientation)
            num_steps: Number of steps to divide the trajectory into
            initial_q: Initial joint configuration for first IK (uses current if None)
        Returns:
            q_trajectory: List of joint angle configurations for each step
            success: True if all IK solutions converged, False otherwise
        """
        # Get end effector frame ID
        ee_frame_id = self.pin_model.getFrameId("link06")
        # Get initial configuration if not provided
        if initial_q is None:
            q_current, _ = self.get_state()
            q_current = q_current[: self.num_joints].copy()
        else:
            q_current = initial_q.copy()
        # If no orientation specified, use current orientation
        if orientation is None:
            pin.forwardKinematics(self.pin_model, self.pin_data, q_current)
            pin.updateFramePlacements(self.pin_model, self.pin_data)
            orientation = self.pin_data.oMf[ee_frame_id].rotation.copy()
        # Create linear interpolation between start and target positions
        q_trajectory = []
        success = True
        for step in range(num_steps + 1):
            # Linear interpolation: p = p_start + t * (p_target - p_start)
            t = step / num_steps
            interp_position = start_position + t * (target_position - start_position)
            # Solve IK for this intermediate position
            q_sol, step_success = self.solve_ik(
                interp_position,
                target_orientation=orientation,
                initial_q=q_current,  # Use previous solution as initial guess
                max_iter=100,
                eps=1e-4,
                orientation_weight=1.0,
            )
            if not step_success:
                print(f"IK failed at step {step}/{num_steps}")
                success = False
                # We'll still include this solution and continue
            # Add solution to trajectory
            q_trajectory.append(q_sol)
            # Update current q for next iteration
            q_current = q_sol.copy()
        return q_trajectory, success

    def execute_trajectory(self, q_trajectory, duration=3.0):
        """
        Execute a trajectory of joint positions with timing
        Args:
            q_trajectory: List of joint angle configurations to execute
            duration: Total duration for the trajectory in seconds
        """
        num_points = len(q_trajectory)
        time_per_point = duration / (num_points - 1) if num_points > 1 else duration
        for i, q in enumerate(q_trajectory):
            start_time = time.time()
            # Execute this configuration
            self.send_position(q)
            # Wait for the right amount of time
            elapsed = time.time() - start_time
            if elapsed < time_per_point:
                time.sleep(time_per_point - elapsed)


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
    while time.time() - start_time < 4:
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
    # Example usage for moving 10cm along x axis while maintaining orientation:
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
    # Get current end effector position and orientation
    current_ee_pos = sim.pin_data.oMf[ee_frame_id].translation
    current_ee_orientation = sim.pin_data.oMf[ee_frame_id].rotation
    print(f"Current end effector position: {current_ee_pos}")
    # Target position: 10cm along X axis from current position
    target_position = np.array([0.45, 0, 0.05])
    print(f"Target end effector position: {target_position}")
    # Solve IK with orientation constraint
    q_solution, success = sim.solve_ik(
        target_position,
        target_orientation=current_ee_orientation,  # Keep current orientation
        initial_q=q_init,
        orientation_weight=1.0,  # Equal priority to position and orientation
    )
    if success:
        print(f"IK solved successfully. Joint angles: {q_solution}")
        # Move to the solution position
        start_time = time.time()
        while time.time() - start_time < 3:  # 3 seconds of movement
            sim.send_position(q_solution)
            viewer.sync()
            time.sleep(0.01)
        # Verify final position and orientation
        q_final, _ = sim.get_state()
        pin.forwardKinematics(sim.pin_model, sim.pin_data, q_final[: sim.num_joints])
        pin.updateFramePlacements(sim.pin_model, sim.pin_data)
        final_ee_pos = sim.pin_data.oMf[ee_frame_id].translation
        final_ee_orientation = sim.pin_data.oMf[ee_frame_id].rotation
        print(f"Final end effector position: {final_ee_pos}")
        print(
            f"Position error: {np.linalg.norm(final_ee_pos - target_position):.6f} meters"
        )
        # Calculate orientation error
        orientation_error = pin.log3(current_ee_orientation @ final_ee_orientation.T)
        print(f"Orientation error: {np.linalg.norm(orientation_error):.6f} radians")
    else:
        print("IK did not converge to a solution")
    # Keep viewer open for a few seconds
    time.sleep(3)
    viewer.close()
