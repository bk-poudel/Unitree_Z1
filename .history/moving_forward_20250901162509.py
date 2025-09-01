import mujoco
import mujoco.viewer as viewer
import numpy as np
from scipy.interpolate import interp1d
import time


def pd_cartesian_torque(model, data, x_des, x_dot_des, kp=200, kd=20):
    """
    Compute joint torques to track a Cartesian target using a Jacobian transpose PD controller.
    This function calculates the position and velocity error of the end-effector in Cartesian
    space, and then uses the Jacobian transpose to project this error back to joint torques.
    This method provides an approximate solution to the inverse dynamics problem.
    Args:
        model (mujoco.MjModel): The MuJoCo model.
        data (mujoco.MjData): The MuJoCo data.
        x_des (np.ndarray): Desired end-effector position in 3D space.
        x_dot_des (np.ndarray): Desired end-effector velocity in 3D space.
        kp (float): Proportional gain for position error.
        kd (float): Derivative gain for velocity error.
    Returns:
        np.ndarray: Computed joint torques to be applied to the robot.
    """
    # Get the ID of the end-effector body from the model.
    # We use a body named 'gripperMover' as defined in the accompanying scene.xml file.
    ee_body = model.body("gripperMover")
    # Get the current position of the end-effector.
    x_current = data.body(ee_body.id).xpos
    # Compute the position Jacobian for the end-effector.
    # The Jacobian (J) maps joint velocities to Cartesian velocities (v = J * q_dot).
    # mj_jacBody computes both the position (jacp) and orientation (jacr) Jacobian.
    J_pos = np.zeros((3, model.nv))
    mujoco.mj_jacBody(model, data, J_pos, None, ee_body.id)
    # Calculate the current end-effector velocity using the Jacobian.
    x_dot_current = J_pos @ data.qvel
    # Calculate the Cartesian PD error.
    e_pos = x_des - x_current
    e_vel = x_dot_des - x_dot_current
    # Compute the desired Cartesian force/torque based on the error.
    # F_des = Kp * e_pos + Kd * e_vel
    cartesian_force = kp * e_pos + kd * e_vel
    # Project the desired Cartesian force back to joint torques using the Jacobian transpose.
    # This is a common and simple method for force control.
    tau = J_pos.T @ cartesian_force
    # The resulting torques correspond to the actuated joints.
    return tau


def create_trajectory(p0, p1, total_time=6.0, n_points=2000):
    """
    Create a linear Cartesian trajectory and return an interpolator for position.
    Args:
        p0 (np.ndarray): Start position (3,).
        p1 (np.ndarray): End position (3,).
        total_time (float): Total time for the trajectory.
        n_points (int): Number of points to sample for the interpolator.
    Returns:
        tuple: A tuple containing:
               - traj_pos_interp (interp1d): An interpolator to get the desired position at any time.
               - x_dot_des (np.ndarray): The constant desired velocity for the linear path.
    """
    # Create an array of positions linearly spaced between p0 and p1.
    trajectory = np.linspace(p0, p1, n_points)
    # Create an array of time points corresponding to the trajectory.
    time_points = np.linspace(0, total_time, n_points)
    # Create a linear interpolator for the trajectory.
    # The interpolator allows us to get the desired position for any time t.
    traj_pos_interp = interp1d(
        time_points, trajectory.T, kind="linear", fill_value="extrapolate"
    )
    # Calculate the constant desired velocity for the linear trajectory.
    x_dot_des = (p1 - p0) / total_time
    return traj_pos_interp, x_dot_des


def main():
    """
    Main function to load the model, create the trajectory, and run the simulation loop.
    """
    # Load the MuJoCo model from the XML file.
    model = mujoco.MjModel.from_xml_path("./scene.xml")
    data = mujoco.MjData(model)
    # Launch the passive MuJoCo viewer.
    viewer_handle = viewer.launch_passive(model, data)
    # Wait for a moment to allow the viewer to initialize.
    time.sleep(1)
    # Define the Cartesian trajectory.
    ee_body = model.body("gripperMover")
    p0 = data.body(ee_body.id).xpos
    # Start position (x, y, z)
    # The new end position is 0.4 units ahead in the x-direction.
    p1 = np.array([0.4, 0, 0.15])  # New end position (x, y, z)
    total_time = 5.0  # Time to complete the trajectory
    traj_pos_interp, x_dot_des = create_trajectory(p0, p1, total_time)
    # Define the simulation parameters.
    sim_time = 0.0
    dt = model.opt.timestep
    try:
        # Main simulation loop. The loop now stops when the trajectory is complete.
        while viewer_handle.is_running() and sim_time < total_time:
            # Get the desired position at the current simulation time.
            x_des = traj_pos_interp(sim_time)
            # Compute the joint torques using the PD controller.
            tau = pd_cartesian_torque(model, data, x_des, x_dot_des)
            # Apply the computed torques to the robot's actuators.
            # The number of actuators in scene.xml should match the number of joints.
            data.ctrl[:] = tau
            # Step the simulation forward by one timestep.
            mujoco.mj_step(model, data)
            # Synchronize the viewer with the updated simulation data.
            viewer_handle.sync()
            # Update the simulation time.
            sim_time += dt
            # Log the progress every 0.2 seconds (200 steps at 1ms timestep).
            if int(sim_time / dt) % 200 == 0:
                ee_body_id = model.body("gripperMover").id
                x_current = data.body(ee_body_id).xpos
                print(f"t={sim_time:.2f}s, EE pos: {x_current}, target: {x_des}")
    except KeyboardInterrupt:
        print("\nTrajectory tracking stopped by user.")
    finally:
        # Print a message to confirm the simulation has ended.
        print("\nTrajectory complete. Simulation has ended.")
        # Clean up by closing the viewer.
        viewer_handle.close()


if __name__ == "__main__":
    main()
