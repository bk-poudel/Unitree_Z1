import mujoco
import mujoco.viewer as viewer
import numpy as np
from scipy.interpolate import interp1d
import time


# ----------------------------
# PD control in Cartesian space
# ----------------------------
def pd_cartesian_torque(model, data, joint_idxs, x_des, x_dot_des, kp=100, kd=10):
    """
    Compute joint torques to track Cartesian target using Jacobian transpose PD.
    joint_idxs: list of controlled joints
    x_des: desired EE position (3,)
    x_dot_des: desired EE velocity (3,)
    """
    # Get end-effector body ID
    ee_body = model.body("gripperMover")  # <-- replace 'ee' with your EE body name
    # Current EE position
    x_current = mujoco.mj_body_xpos(model, data, ee_body.id)
    # Compute Jacobian for EE
    J = np.zeros((3, model.nv))
    mujoco.mj_jacBody(model, data, J, None, ee_body.id)
    # Current EE velocity
    x_dot_current = J @ data.qvel
    # Cartesian PD error
    e_pos = x_des - x_current
    e_vel = x_dot_des - x_dot_current
    # Torque via Jacobian transpose
    tau = J.T @ (kp * e_pos + kd * e_vel)
    # Only apply to controlled joints
    tau_full = np.zeros(model.nu)
    tau_full[joint_idxs] = tau[joint_idxs]
    return tau_full


# ----------------------------
# Trajectory definition
# ----------------------------
def create_trajectory(p0, p1, total_time=5.0, n_points=100):
    """
    Create linear Cartesian trajectory and return interpolator.
    """
    trajectory = np.linspace(p0, p1, n_points)
    time_points = np.linspace(0, total_time, n_points)
    traj_interp = interp1d(
        time_points, trajectory.T, kind="linear", fill_value="extrapolate"
    )
    return traj_interp, total_time


# ----------------------------
# Main function
# ----------------------------
def main():
    # Load model
    model = mujoco.MjModel.from_xml_path("./scene.xml")
    data = mujoco.MjData(model)
    # Launch viewer
    viewer_handle = viewer.launch_passive(model, data)
    time.sleep(1)
    # Define Cartesian trajectory
    p0 = np.array([0.5, 0.0, 0.5])  # start position
    p1 = np.array([0.5, 0.3, 0.7])  # end position
    traj_interp, total_time = create_trajectory(p0, p1, total_time=5.0, n_points=100)
    # PD gains
    kp = 200
    kd = 20
    look_ahead_time = 0.2  # seconds ahead
    # Controlled joints (all actuated)
    joint_idxs = np.arange(model.nu)
    sim_time = 0.0
    dt = model.opt.timestep  # use model timestep
    try:
        while viewer_handle.is_running():
            # Desired EE position and velocity with look-ahead
            x_des = traj_interp(sim_time + look_ahead_time)
            x_dot_des = np.zeros(3)
            # Compute torque
            tau = pd_cartesian_torque(model, data, joint_idxs, x_des, x_dot_des, kp, kd)
            # Apply torque
            data.ctrl[:] = tau
            # Step simulation
            mujoco.mj_step(model, data)
            viewer_handle.sync()
            # Optional logging every 0.1s
            if int(sim_time / dt) % 100 == 0:
                ee_body = model.body("ee")
                x_current = mujoco.mj_body_xpos(model, data, ee_body.id)
                print(f"t={sim_time:.2f}s, EE pos: {x_current}, target: {x_des}")
            sim_time += dt
            time.sleep(dt)
    except KeyboardInterrupt:
        print("\nTrajectory tracking stopped by user")
    finally:
        viewer_handle.close()


if __name__ == "__main__":
    main()
