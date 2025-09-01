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
    # Get end-effector body ID
    ee_body_id = model.body("gripperMover").id
    # Current EE position and velocity
    x_current = data.body(ee_body_id).xpos
    J_pos = np.zeros((3, model.nv))
    mujoco.mj_jacBody(model, data, J_pos, None, ee_body_id)
    x_dot_current = J_pos @ data.qvel
    # Cartesian PD error
    e_pos = x_des - x_current
    e_vel = x_dot_des - x_dot_current
    # Torque via Jacobian transpose. The result has size (model.nv,),
    # but we only have actuators for the first model.nu joints.
    tau_full_dof = J_pos.T @ (kp * e_pos + kd * e_vel)
    # Create an actuator torque vector of the correct size (model.nu).
    # This assumes the first model.nu degrees of freedom are the actuated joints.
    tau = np.zeros(model.nu)
    tau = tau_full_dof[: model.nu]
    # This is the original torque vector for all 7 joints.
    return tau


# ----------------------------
# Trajectory definition
# ----------------------------
def create_trajectory(p0, p1, total_time=6.0, n_points=2000):
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
    x_dot_des = (p1 - p0) / total_time
    return traj_pos_interp, x_dot_des


# ----------------------------
# Main function
# ----------------------------
def main():
    # Load model
    model = mujoco.MjModel.from_xml_path("./scene.xml")
    # Add print statements to diagnose the qpos size issue
    print(f"Number of degrees of freedom (model.nq): {model.nq}")
    print(f"Number of actuators (model.nu): {model.nu}")
    data = mujoco.MjData(model)
    # Launch viewer
    viewer_handle = viewer.launch_passive(model, data)
    # Give viewer time to initialize
    time.sleep(1)
    # Define Cartesian trajectory
    p0 = data.body(
        "gripperMover"
    ).xpos.copy()  # Use the initial position of the end effector
    p1 = p0 + np.array([0.4, 0, 0])  # end position 0.4m ahead in x
    traj_pos_interp, x_dot_des = create_trajectory(p0, p1, total_time=5.0)
    # PD gains
    kp = 200
    kd = 20
    sim_time = 0.0
    dt = model.opt.timestep  # use model timestep
    try:
        while viewer_handle.is_running() and sim_time < 5.0:
            # Desired EE position based on simulation time
            x_des = traj_pos_interp(sim_time)
            # Compute torque
            tau = pd_cartesian_torque(model, data, x_des, x_dot_des, kp, kd)
            # --- MODIFIED CODE START ---
            # Create a zeroed control vector
            ctrl_vector = np.zeros(model.nu)
            # Apply the calculated torque ONLY to the 7th joint (index 6)
            if model.nu >= 7:
                ctrl_vector[6] = tau[6]
            else:
                print("Warning: The model does not have a 7th actuator to control.")
            # Apply the new control vector
            data.ctrl[:] = ctrl_vector
            # --- MODIFIED CODE END ---
            # Step simulation
            mujoco.mj_step(model, data)
            # Synchronize the viewer with the simulation data
            viewer_handle.sync()
            sim_time += dt
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
        print("Simulation completed.")


if __name__ == "__main__":
    main()
