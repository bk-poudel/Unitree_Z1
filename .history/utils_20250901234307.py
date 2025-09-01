import numpy as np
from scipy.spatial.transform import Rotation as R
import mujoco
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper

END_EFF_FRAME_ID = 15  # Frame ID for the end-effector in the Pinocchio model


def print_model_info(robot):
    """Prints the names and IDs of all bodies and frames in the model."""
    print("--- Bodies ---")
    for i, name in enumerate(robot.model.names):
        print(f"ID: {i}, Name: {name}")
    print("\n--- Frames ---")
    for i, frame in enumerate(robot.model.frames):
        print(f"ID: {i}, Name: {frame.name}")


def get_gravity(robot, q):
    """Computes the gravity vector for a given joint configuration."""
    return pin.computeGeneralizedGravity(robot.model, robot.data, q)


def get_jacobian(robot, q):
    """Computes the Jacobian of the end-effector."""
    pin.computeFrameJacobian(robot.model, robot.data, q, END_EFF_FRAME_ID)
    J_temp = pin.getFrameJacobian(
        robot.model,
        robot.data,
        END_EFF_FRAME_ID,
        pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
    )
    J = np.zeros([6, robot.model.nv])
    J[3:6, :] = J_temp[0:3, :]
    J[0:3, :] = J_temp[3:6, :]
    return J


def get_idx_of_ee(pin_model, name="gripperMover"):
    """
    Returns the index of the end-effector body in the MuJoCo model.
    """
    try:
        idx = pin_model.getJointId(name)
        return idx
    except Exception:
        print(f"Error: Body with name '{name}' not found in the model.")
        return -1


def print_joint_names(model):
    """
    Prints the names of all joints in the loaded MuJoCo model.
    """
    for i in range(model.njnt):
        print(f"Joint {i}: {model.joint(i).name}")


def print_body_names(model):
    """
    Prints the names of all bodies in the loaded MuJoCo model.
    """
    for i in range(model.nbody):
        print(f"Body {i}: {model.body(i).name}")


def print_actuator_names(model):
    """
    Prints the names of all actuators in the loaded MuJoCo model.
    """
    for i in range(model.nu):
        print(f"Actuator {i}: {model.actuator(i).name}")


def get_states(data, model):
    """
    Returns the current joint positions (qpos) and velocities (qvel).
    """
    return data.qpos[: model.nv], data.qvel[: model.nv]


def get_ee_pose(robot, q):
    """
    Retrieves the position and rotation matrix of the end-effector.
    """
    pin.forwardKinematics(robot.model, robot.data, q)
    pin.updateFramePlacements(robot.model, robot.data)
    T_S_F = robot.data.oMf[END_EFF_FRAME_ID]
    return T_S_F.homogeneous


def send_torques(model, data, tau, viewer):
    data.ctrl[: tau.shape[0]] = tau
    mujoco.step(model, data)
    if viewer:
        viewer.sync()
