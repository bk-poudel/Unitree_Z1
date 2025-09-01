import mujoco
import mujoco.viewer as viewer
import numpy as np
from scipy.interpolate import interp1d
import time


# ----------------------------
# PD control in Cartesian space
# ----------------------------
def pd_cartesian_torque(
    model,
    data,
    x_des,
    x_dot_des,
    q_des,
    w_des,
    kp_pos=200,
    kd_pos=20,
    kp_rot=50,
    kd_rot=5,
):
    """
    Compute joint torques to track Cartesian position and orientation targets using
    a single 6D spatial error.
    Args:
        model (mujoco.MjModel): The MuJoCo model.
        data (mujoco.MjData): The MuJoCo data.
        x_des (np.ndarray): Desired end-effector position (3,).
        x_dot_des (np.ndarray): Desired end-effector velocity (3,).
        q_des (np.ndarray): Desired end-effector orientation (quaternion 4,).
        w_des (np.ndarray): Desired end-effector angular velocity (3,).
        kp_pos (float): Proportional gain for position.
        kd_pos (float): Derivative gain for position.
        kp_rot (float): Proportional gain for orientation.
        kd_rot (float): Derivative gain for orientation.
    Returns:
        np.ndarray: Computed joint torques.
    """
    # Get end-effector body ID and name (assuming it's 'gripperMover')
    ee_body = model.body("gripperMover")
    # Current EE position, orientation, and velocities
    x_current = data.body(ee_body.id).xpos
    q_current = data.body(ee_body.id).xquat
    # Get the 6xnv Jacobian that combines position and rotation
    J_pos = np.zeros((3, model.nv))
    J_rot = np.zeros((3, model.nv))
    mujoco.mj_jacBody(model, data, J_pos, J_rot, ee_body.id)
    J_spatial = np.vstack((J_pos, J_rot))
    x_dot_current = J_pos @ data.qvel
    w_current = J_rot @ data.qvel
    # Compute 6D spatial error vector
    e_pos = x_des - x_current
    quat_error = np.zeros(3)
    mujoco.mju_quatSub(quat_error, q_des, q_current)
    spatial_error = np.hstack((e_pos, quat_error))
    # Compute 6D spatial velocity error vector
    v_error = x_dot_des - x_dot_current
    w_error = w_des - w_current
    spatial_vel_error = np.hstack((v_error, w_error))
    # Construct the diagonal gain matrices
    kp_gains = np.diag([kp_pos, kp_pos, kp_pos, kp_rot, kp_rot, kp_rot])
    kd_gains = np.diag([kd_pos, kd_pos, kd_pos, kd_rot, kd_rot, kd_rot])
    # Compute the overall spatial force/torque vector
    spatial_force = kp_gains @ spatial_error + kd_gains @ spatial_vel_error
    # Total torque is the Jacobian transpose multiplied by the spatial force
    tau = J_spatial.T @ spatial_force
    # Return torque for all actuated joints
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
    x_dot_des = (p1 - p0) / total_time
    return traj_pos_interp, x_dot_des


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
    # Set the desired EE orientation to the initial orientation
    ee_body_id = model.body("gripperMover").id
    orientation_desired = data.body(ee_body_id).xmat.copy()
    # Define Cartesian position trajectory
    p0 = data.body(ee_body_id).xpos.copy()
    p1 = np.array([0.3, 0.3, 0.3])  # end position
    traj_pos_interp, x_dot_des = create_trajectory(p0, p1, total_time=5.0)
    # PD gains
    kp_pos = 200
    kd_pos = 20
    kp_rot = 50
    kd_rot = 5
    sim_time = 0.0
    dt = model.opt.timestep  # use model timestep
    total_time = 5.0
    try:
        while viewer_handle.is_running():
            # Clamp the simulation time to the total trajectory time
            clamped_sim_time = min(sim_time, total_time)
            # Desired EE position and velocity based on simulation time
            x_des = traj_pos_interp(clamped_sim_time)
            # The desired velocity should be zero after the trajectory is complete
            if clamped_sim_time >= total_time:
                x_dot_des = np.zeros(3)
            # Compute torque
            tau = pd_cartesian_torque(
                model,
                data,
                x_des,
                x_dot_des,
                q_des,
                w_des,
                kp_pos=kp_pos,
                kd_pos=kd_pos,
                kp_rot=kp_rot,
                kd_rot=kd_rot,
            )
            # Apply torque to all actuated joints
            data.ctrl[:] = tau
            # Step simulation
            mujoco.mj_step(model, data)
            # Synchronize the viewer with the simulation data
            viewer_handle.sync()
            sim_time += dt
            # Optional logging every 0.2s
            if int(sim_time / dt) % 200 == 0:
                x_current = data.body(ee_body_id).xpos
                print(f"t={sim_time:.2f}s, EE pos: {x_current}, target: {x_des}")
    except KeyboardInterrupt:
        print("\nTrajectory tracking stopped by user.")
    finally:
        viewer_handle.close()


if __name__ == "__main__":
    main()
