import pandas as pd
import numpy as np
import termios, tty, sys, select, time
import unitree_arm_interface

# Load poses from cleaned CSV
pose_data = {
    "pick_bottle_top": [-0.127113, 2.930545, -2.6075, 1.273424, -0.007536, -1.791614],
    "pick_Gbox_final_side": [-0.435, 2.95, -2.32, 1.19, 0.0, -1.85],
    "drop_final": [-0.34, 2.80, -2.45, 1.25, 0.0, -1.90]
}

pose_map = {
    "1": "pick_bottle_top",
    "2": "pick_Gbox_final_side",
    "3": "drop_final"
}

def get_key(timeout=0.1):
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        rlist, _, _ = select.select([fd], [], [], timeout)
        if rlist:
            key = sys.stdin.read(1)
        else:
            key = ''
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return key

def move_to_pose(targetPos):
    arm = unitree_arm_interface.ArmInterface(hasGripper=True)
    armModel = arm._ctrlComp.armModel
    arm.setFsmLowcmd()

    duration = 1000
    lastPos = arm.lowstate.getQ()

    for i in range(duration):
        alpha = i / duration
        arm.q = (1 - alpha) * lastPos + alpha * np.array(targetPos)
        arm.qd = (np.array(targetPos) - lastPos) / (duration * 0.002)
        arm.tau = armModel.inverseDynamics(arm.q, arm.qd, np.zeros(6), np.zeros(6))
        arm.setArmCmd(arm.q, arm.qd, arm.tau)
        arm.setGripperCmd(0.02, 0.0, 0.0)
        arm.sendRecv()
        time.sleep(arm._ctrlComp.dt)

    arm.loopOn()
    arm.backToStart()
    arm.loopOff()

print("Press 1, 2, or 3 to move to a saved pose:")
print("1 - pick_bottle_top\n2 - pick_Gbox_final_side\n3 - drop_final")
print("Press 'q' to quit.")

while True:
    key = get_key()
    if key == 'q':
        break
    elif key in pose_map:
        pose_name = pose_map[key]
        print(f"Moving to: {pose_name}")
        move_to_pose(pose_data[pose_name])
