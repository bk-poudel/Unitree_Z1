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


def inverse_kinematics(model, data, ee_site_name, target_pos, max_iter=100, tol=1e-3):
    """Simple inverse kinematics using Jacobian transpose (for demonstration)."""
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, ee_site_name)
    for _ in range(max_iter):
        mujoco.mj_forward(model, data)
        ee_pos = data.site_xpos[site_id][:3]
        error = target_pos - ee_pos
        if np.linalg.norm(error) < tol:
            break
        jacp = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacp, None, site_id)
        dq = jacp.T @ error * 0.1  # step size
        data.qpos[: model.nv] += dq
        data.qpos[:] = np.clip(data.qpos, model.jnt_range[:, 0], model.jnt_range[:, 1])
    mujoco.mj_forward(model, data)


def main():
    # Load the model from XML
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    # Create a data object for simulation
    data = mujoco.MjData(model)
    # Parameters for the square
    center = [0.0, 0.0, 0.2]  # XY plane at Z=0.2
    size = 0.3
    steps_per_side = 40
    square_path = generate_square_path(center[:2], size, steps_per_side)
    # End effector site name (update to match your robot's XML)
    ee_site_name = "ee_site"
    # Launch the viewer for visualization
    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("Robot end effector drawing square in XY plane. Press ESC to exit.")
        for xy in square_path:
            target = np.array([xy[0], xy[1], center[2]])
            inverse_kinematics(model, data, ee_site_name, target)
            mujoco.mj_step(model, data)
            viewer.sync()


if __name__ == "__main__":
    main()
