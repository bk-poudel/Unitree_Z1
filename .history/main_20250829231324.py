import mujoco
import mujoco.viewer
import numpy as np
import os

# Path to your scene.xml file
XML_PATH = os.path.join(os.path.dirname(__file__), "scene.xml")


def generate_square_path(center, size, steps_per_side):
    """Generate (x, y) points for a square in the XY plane."""
    half = size / 2
    corners = [
        (center[0] - half, center[1] - half),
        (center[0] + half, center[1] - half),
        (center[0] + half, center[1] + half),
        (center[0] - half, center[1] + half),
        (center[0] - half, center[1] - half),  # Close the square
    ]
    path = []
    for i in range(4):
        start = np.array(corners[i])
        end = np.array(corners[i + 1])
        for t in np.linspace(0, 1, steps_per_side, endpoint=False):
            point = start + t * (end - start)
            path.append(point)
    return np.array(path)


def get_end_effector_body_id(model):
    # Assumes the last body is the end effector (update if needed)
    return model.nbody - 1


def get_end_effector_pos(data, body_id):
    return data.xpos[body_id][:3]


def inverse_kinematics(model, data, body_id, target_pos, max_iter=100, tol=1e-3):
    """Simple inverse kinematics using Jacobian transpose (for demonstration)."""
    for _ in range(max_iter):
        mujoco.mj_forward(model, data)
        ee_pos = get_end_effector_pos(data, body_id)
        error = target_pos - ee_pos
        if np.linalg.norm(error) < tol:
            break
        jacp = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data, jacp, None, body_id)
        dq = jacp.T @ error * 0.1  # step size
        data.qpos[: model.nv] += dq
        data.qpos[:] = np.clip(data.qpos, model.jnt_range[:, 0], model.jnt_range[:, 1])
    mujoco.mj_forward(model, data)


def main():
    # Load the model from XML
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    # Move EE in x direction only, 10 cm
    start_pos = [0.0, 0.0, 0.2]
    end_pos = [0.4, 0.0, 0.2]  # 10 cm in x
    steps = 4000
    x_path = np.linspace(start_pos[0], end_pos[0], steps)
    # Get end effector body id
    ee_body_id = get_end_effector_body_id(model)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("Robot end effector moving 10 cm in x direction. Press ESC to exit.")
        for x in x_path:
            target = np.array([x, start_pos[1], start_pos[2]])
            inverse_kinematics(model, data, ee_body_id, target)
            mujoco.mj_step(model, data)
            viewer.sync()


if __name__ == "__main__":
    main()
