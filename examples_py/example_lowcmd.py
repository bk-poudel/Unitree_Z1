import sys

sys.path.append("../lib")
import unitree_arm_interface
import time
import numpy as np


def moving_average(data, window_size=5):
    return np.convolve(data, np.ones(window_size) / window_size, mode="same")


print("Press ctrl+\ to quit process.")
# --- 1. Load Trajectory Data from CSV ---
trajectory_file = "joint_states.csv"
print(f"Loading trajectory from: {trajectory_file}")
try:
    # Load the entire csv file into a numpy array
    trajectory = np.loadtxt(trajectory_file, delimiter=",")
    print(f"Successfully loaded {len(trajectory)} points from trajectory.")
except FileNotFoundError:
    print(f"Error: The file '{trajectory_file}' was not found.")
    print("Please make sure the CSV file is in the same directory as the script.")
    sys.exit(1)
except Exception as e:
    print(f"An error occurred while reading the CSV file: {e}")
    sys.exit(1)
np.set_printoptions(precision=3, suppress=True)
arm = unitree_arm_interface.ArmInterface(hasGripper=True)
armModel = arm._ctrlComp.armModel
trajectory = trajectory[:, :7]
qd_traj = np.gradient(trajectory[:, 0:6], arm._ctrlComp.dt, axis=0)
qd_smoothed = np.zeros_like(qd_traj)
for i in range(6):
    qd_smoothed[:, i] = moving_average(qd_traj[:, i], window_size=5)
qd_traj = qd_smoothed
arm.setFsmLowcmd()
num_trajectory_points = len(trajectory)
for i in range(num_trajectory_points):
    current_q = arm.lowstate.getQ()
    target_q = trajectory[i, 0:6]
    arm.q = target_q  # set position
    arm.qd = qd_traj[i]  # set velocity
    arm.tau = armModel.inverseDynamics(
        arm.q, arm.qd, np.zeros(6), np.zeros(6)
    )  # set torque
    arm.tau = np.clip(arm.tau, -20, 20)
    arm.gripperQ = trajectory[i, 6]
    # arm.setArmCmd(arm.q, arm.qd, arm.tau)
    # arm.setGripperCmd(arm.gripperQ, arm.gripperQd, arm.gripperTau)
    # arm.sendRecv()  # udp connection
    # print(arm.lowstate.getQ())
    time.sleep(arm._ctrlComp.dt)
arm.loopOn()
arm.backToStart()
arm.loopOff()
