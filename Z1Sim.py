import time
from copy import deepcopy
import mujoco
import mujoco.viewer
import numpy as np
import os
import pinocchio as pin
from pinocchio import RobotWrapper
from scipy.spatial.transform import Rotation as R

END_EFF_FRAME_ID = 15


class Z1Sim:
    def __init__(self, xml_path, dt=0.001):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.pin_model = RobotWrapper.BuildFromURDF(
            "/home/bibek/Unitree_Z1/Unitree_Z1/z1.urdf",
            "/home/bibek/Unitree_Z1/Unitree_Z1/z1_description/meshes",
        )
        # Key position
        self.q0 = np.array(
            [-5.06e-12, 0.7844, -0.2552, -0.5213, -9.82e-08, -1.24e-05, 3.56e-05]
        )
        self.data.qpos[:7] = self.q0
        self.model.opt.gravity[2] = -9.81
        self.dt = dt
        self.model.opt.timestep = dt
        # Viewer
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.step_counter = 0
        _render_dt = 1 / 60
        self.render_ds_ratio = max(1, int(_render_dt // dt))
        # Initialize
        mujoco.mj_forward(self.model, self.data)
        self.viewer.sync()
        # State holders
        self.nv = self.model.nv
        self.actuator_tau = np.zeros(6)
        self.tau_ff = np.zeros(6)
        self.dq_des = np.zeros(6)
        self.latest_command_stamp = time.time()

    def reset(self):
        """Reset robot to key position and hold it"""
        self.data.qpos[:7] = self.q0
        self.data.qvel[:7] = np.zeros(7)
        # Apply gravity compensation torques to hold the position
        self.tau_ff = self.compute_gravity_compensation(self.q0)
        # Step a few times to stabilize
        for _ in range(10):
            self.send_joint_torque(self.tau_ff)
            self.viewer.sync()

    def compute_gravity_compensation(self, q):
        """Compute torques to hold current pose"""
        mujoco.mj_forward(self.model, self.data)
        # Gravity + Coriolis (mj_inverse does it internally if needed)
        tau_grav = np.zeros(6)
        mujoco.mj_inverse(self.model, self.data)  # fills data.qfrc_inverse
        tau_grav[:] = self.data.qfrc_inverse[:6]
        return tau_grav

    def get_gravity(self, q):
        g = self.pin_model.gravity(q[:6])
        return g[:6]

    def step(self, finger_pos=None):
        tau = self.tau_ff
        if finger_pos is not None:
            tau = np.append(tau, finger_pos)
            self.data.ctrl[:7] = tau.squeeze()
        else:
            self.data.ctrl[:6] = tau.squeeze()
        self.step_counter += 1
        mujoco.mj_step(self.model, self.data)
        if (self.step_counter % self.render_ds_ratio) == 0:
            self.viewer.sync()

    def send_joint_torque(self, torques, finger_pos=None):
        """Continuously apply torque"""
        self.tau_ff = torques
        self.latest_command_stamp = time.time()
        self.step(finger_pos)

    def get_state(self):
        return self.data.qpos, self.data.qvel

    def forward_kinematics(self, q):
        T_S_F = self.pin_model.framePlacement(
            q, END_EFF_FRAME_ID, update_kinematics=True
        )
        return np.array(T_S_F)

    def get_jacobian(self, q):
        J_temp = self.pin_model.computeFrameJacobian((q[:6]), END_EFF_FRAME_ID)
        J = np.zeros([6, 6])
        J[3:6, :] = J_temp[0:3, :6]
        J[0:3, :] = J_temp[3:6, :6]
        return J[:, :6]

    def close(self):
        self.viewer.close()
