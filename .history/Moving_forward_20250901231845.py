import mujoco
import numpy as np
import mujoco.viewer as viewer
import time
from utils import *
# Simulation parameters
dt = 0.001  # 1khz frequency
model = mujoco.MjModel.from_xml_path("./scene.xml")
data = mujoco.MjData(model)
robot = RobotWrapper.BuildFromURDF(
    "./z1.urdf", "/home/bibek/Unitree_Z1/Unitree_Z1/z1_description/meshes"
)
home_pos = np.array([0, 0.785, -0.261, -0.523, 0, 0, 0])
data.qpos[:7] = home_pos
mujoco.mj_forward(model, data)
    q, dq = get_states(data)
viewer = viewer.launch_passive(model, data)
xpos = get_ee_pose(q, robot)[:3, 3]
print_model_info(robot)
while True:
    q, dq = get_states(data)
    xpos = get_ee_pose(q, robot)[:3, 3]
    orientation = get_ee_pose(q, robot)[:3, :3]
    print(f"q: {q}, dq: {dq}")
    # Step the simulation
    mujoco.mj_step(model, data)
    print(f"End Effector Pose: {get_ee_pose(q, robot)} \n")
    print(f"Joint States: {get_states(data)}\n")
    viewer.sync()
    time.sleep(dt)
