import mujoco
import mujoco.viewer
import numpy as np


class Z1sim:
    def __init__(self, xml_path="scene.xml"):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.viewer = None

    def launch_viewer(self):
        mujoco.viewer.launch(self.model, self.data)

    def set_joint_positions(self, joint_names, positions):
        """
        Set the positions of specified joints.
        joint_names: list of joint names (str)
        positions: list or np.array of positions (float)
        """
        for name, pos in zip(joint_names, positions):
            idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if idx == -1:
                print(f"Joint {name} not found.")
                continue
            self.data.qpos[idx] = pos

    def step(self):
        mujoco.mj_step(self.model, self.data)

    def set_joint_velocities(self, joint_names, velocities):
        """
        Set the velocities of specified joints.
        joint_names: list of joint names (str)
        velocities: list or np.array of velocities (float)
        """
        for name, vel in zip(joint_names, velocities):
            idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if idx == -1:
                print(f"Joint {name} not found.")
                continue
            self.data.qvel[idx] = vel

    def apply_joint_torque(self, joint_names, torques):
        """
        Apply torques to specified joints.
        joint_names: list of joint names (str)
        torques: list or np.array of torques (float)
        """
        for name, torque in zip(joint_names, torques):
            idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if idx == -1:
                print(f"Joint {name} not found.")
                continue
            self.data.ctrl[idx] = torque

    def reset(self):
        self.data = mujoco.MjData(self.model)


# Example usage:
# sim = Z1sim("scene.xml")
# sim.set_joint_positions(["joint1", "joint2"], [0.5, -0.5])
#
