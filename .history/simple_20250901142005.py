import mujoco
import pinocchio as pin
import numpy as np
import mujoco.viewer as viewer
import time

model = mujoco.MjModel.from_xml_path("./scene.xml")
data = mujoco.MjData(model)
dt = 0.001
home_pos = np.array([0, 0.785, -0.261, -0.523, 0, 0])
data.qpos[:6] = home_pos
model.opt.gravity[:] = 0
print("Joint names:")
for i in range(model.njnt):
    print(model.joint(i).name)
print("\nActuator names:")
for i in range(model.nu):
    print(model.actuator(i).name)
print("\nBody names:")
for i in range(model.nbody):
    print(model.body(i).name)
mujoco.mj_step(model, data)
viewer = viewer.launch_passive(model, data)
while True:
    data.qfrc_applied[:6] = np.array([0.6, 0, 0, 0, 0, 0])  # direct torque
    mujoco.mj_step(model, data)
    viewer.sync()
    time.sleep(dt)
