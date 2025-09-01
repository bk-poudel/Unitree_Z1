import mujoco
import mujoco.viewer
import numpy as np
import pinocchio as pin


class Z1sim:
    def __init__(self, xml_path="scene.xml", urdf_path="scene.urdf"):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.viewer = None
        # Load Pinocchio model for dynamics
        self.pin_model = pin.buildModelFromUrdf(urdf_path)
        self.pin_data = self.pin_model.createData()

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

    def get_dynamics(self, q, v):
        """
        Get the mass matrix and nonlinear effects using Pinocchio.
        q: joint positions (numpy array)
        v: joint velocities (numpy array)
        Returns: mass matrix (M), nonlinear effects (b)
        """
        pin.computeAllTerms(self.pin_model, self.pin_data, q, v)
        M = pin.crba(self.pin_model, self.pin_data, q)
        b = pin.rnea(self.pin_model, self.pin_data, q, v, np.zeros_like(v))
        return M, b


# Example usage:
if __name__ == "__main__":
    sim = Z1sim("scene.xml", "scene.urdf")
    sim.launch_viewer()
    while True:
        sim.apply_joint_torque(["joint1", "joint2"], [10, -0.5])
        sim.step()
        # Example: get dynamics
        q = np.array([sim.data.qpos[0], sim.data.qpos[1]])
        v = np.array([sim.data.qvel[0], sim.data.qvel[1]])
        M, b = sim.get_dynamics(q, v)
        print("Mass matrix:", M)
