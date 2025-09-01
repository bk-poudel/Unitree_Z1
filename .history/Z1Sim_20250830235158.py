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
        self.data.ctrl[:6] = tau
        mujoco.mj_step(self.model, self.data)

    def send_constant_torque(self, torque_command):
        """
        Applies constant torque to joints with gravity compensation

        Args:
            torque_command: Array of desired torques for each joint
        """
        # Get current state
        q, qdot = self.get_state()

        # Compute gravity compensation torques using Pinocchio
        gravity_torques = np.zeros(self.pin_model.nv)
        pin.computeGeneralizedGravity(self.pin_model, self.pin_data, q)
        gravity_torques = self.pin_data.g

        # Add gravity compensation to your desired torque
        total_torque = torque_command + gravity_torques[:len(torque_command)]

        # Apply the total torque
        self.data.ctrl[:len(total_torque)] = total_torque
        mujoco.mj_step(self.model, self.data)


# Example usage:
mjcf_path = "scene.xml"
urdf_path = "z1.urdf"
sim = Z1Sim(mjcf_path, urdf_path)
sim.reset()
viewer = mujoco.viewer.launch_passive(sim.model, sim.data)
start_time = time.time()
while time.time() - start_time < 10:
    sim.send_torque(np.array([0.001, 0, 0, 0, 0, 0]))
    viewer.sync()
viewer.close()
