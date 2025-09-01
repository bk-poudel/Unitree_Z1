import mujoco
import mujoco.viewer as viewer
import numpy as np
from scipy.interpolate import interp1d
import time


# ----------------------------
# PD control in Cartesian space
# ----------------------------
def pd_cartesian_torque(model, data, x_des, x_dot_des, kp=200, kd=20):
    """
    Compute joint torques to track Cartesian target using Jacobian transpose PD.
    Args:
        model (mujoco.MjModel): The MuJoCo model.
        data (mujoco.MjData): The MuJoCo data.
        x_des (np.ndarray): Desired end-effector position (3,).
        x_dot_des (np.ndarray): Desired end-effector velocity (3,).
        kp (float): Proportional gain.
        kd (float): Derivative gain.
    Returns:
        np.ndarray: Computed joint torques.
    """
    # Get end-effector body ID and name (assuming it's 'gripperMover')
    ee_body = model.body("gripperMover")
    # Current EE position using the correct attribute access
    x_current = data.body(ee_body.id).xpos
    # Compute Jacobian for EE. We only need the position Jacobian (jacp).
    J_pos = np.zeros((3, model.nv))
    mujoco.mj_jacBody(model, data, J_pos, None, ee_body.id)
    # Current EE velocity
    x_dot_current = J_pos @ data.qvel
    # Cartesian PD error
    e_pos = x_des - x_current
    e_vel = x_dot_des - x_dot_current
    # Torque via Jacobian transpose
    tau = J_pos.T @ (kp * e_pos + kd * e_vel)
    # Apply to all controlled joints
    return tau


# ----------------------------
# Trajectory definition
# ----------------------------
def create_trajectory(p0, p1, total_time=5.0, n_points=100):
    """
    Create a linear Cartesian trajectory and return interpolator.
    Args:
        p0 (np.ndarray): Start position (3,).
        p1 (np.ndarray): End position (3,).
        total_time (float): Total time for the trajectory.
        n_points (int): Number of points in the trajectory.
    Returns:
        tuple: A tuple containing the position interpolator and the desired velocity.
    """
    trajectory = np.linspace(p0, p1, n_points)
    time_points = np.linspace(0, total_time, n_points)
    # Interpolator for position
    traj_pos_interp = interp1d(
        time_points, trajectory.T, kind="linear", fill_value="extrapolate"
    )
    # Desired velocity is constant for a linear trajectory
    return traj_pos_interp


# ----------------------------
# Main function
# ----------------------------
def main():
    # Load model
    model = mujoco.MjModel.from_xml_path("./scene.xml")
    data = mujoco.MjData(model)
    # Reset data to the initial state
    mujoco.mj_resetData(model, data)
    # Launch viewer
    viewer_handle = viewer.launch_passive(model, data)
    # Give viewer time to initialize
    time.sleep(1)
    # Define Cartesian trajectory
    # Set the starting position to the robot's current EE position
    p0 = data.body(model.body("gripperMover").id).xpos.copy()
    p1 = np.array([0.3, 0.3, 0.3])  # end position
    traj_pos_interp = create_trajectory(p0, p1, total_time=5.0)
    x_dot_des = np.zeros(3)  # desired velocity is zero
    # PD gains
    kp = 200
    kd = 20
    dt = model.opt.timestep  # use model timestep
    try:
        for sim_time in len(traj_pos_interp):
            # Desired EE position based on simulation time
            x_des = traj_pos_interp(sim_time)
            # Compute torque
            tau = pd_cartesian_torque(model, data, x_des, x_dot_des, kp, kd)
            # Apply torque to all actuated joints
            data.ctrl[:] = tau
            # Step simulation
            mujoco.mj_step(model, data)
            # Synchronize the viewer with the simulation data
            viewer_handle.sync()
            # Optional logging every 0.2s
            if int(sim_time / dt) % 200 == 0:
                # Use the consistent end-effector name 'gripperMover'
                ee_body_id = model.body("gripperMover").id
                x_current = data.body(ee_body_id).xpos
                print(f"t={sim_time:.2f}s, EE pos: {x_current}, target: {x_des}")
    except KeyboardInterrupt:
        print("\nTrajectory tracking stopped by user.")
    finally:
        viewer_handle.close()


if __name__ == "__main__":
    main()
