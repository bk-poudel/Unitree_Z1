import time
from copy import deepcopy
import mujoco
import mujoco.viewer
import numpy as np
import os
import pinocchio as pin
from pinocchio import RobotWrapper
from scipy.spatial.transform import Rotation as R
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_PATH = os.path.join(SCRIPT_DIR, "assets")
END_EFF_FRAME_ID = 6  # Z1 has 6 joints, so end-effector frame ID is 6


class Z1Sim:
    def __init__(self, interface_type="torque", render=True, dt=0.001, xml_path=None):
        assert interface_type in ["torque"], "The interface should be torque"
        self.interface_type = interface_type
        if xml_path is not None:
            self.model = mujoco.MjModel.from_xml_path(xml_path)
        else:
            # Look for scene.xml in the script directory
            scene_path = os.path.join(SCRIPT_DIR, "scene.xml")
            if not os.path.exists(scene_path):
                raise FileNotFoundError(f"scene.xml not found at {scene_path}")
            self.model = mujoco.MjModel.from_xml_path(scene_path)
        self.simulated = True
        self.data = mujoco.MjData(self.model)
        self.dt = dt
        _render_dt = 1 / 60
        self.render_ds_ratio = max(1, _render_dt // dt)
        if render:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.render = True
            self.viewer.cam.distance = 2.0
            self.viewer.cam.azimuth = 90
            self.viewer.cam.elevation = -20
            self.viewer.cam.lookat[:] = np.array([0.0, 0.0, 0.3])
        else:
            self.render = False
        self.model.opt.gravity[2] = -9.81
        self.model.opt.timestep = dt
        self.step_counter = 0
        self.joint_names = [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
        ]
        # Set the home pose (reset position) for the Z1 robot
        # Based on Z1 joint limits and neutral position
        self.q0 = np.array(
            [
                0.0,  # joint1: base rotation
                1.0,  # joint2: shoulder pitch
                -1.5,  # joint3: elbow pitch
                0.0,  # joint4: wrist pitch
                0.0,  # joint5: wrist roll
                0.0,  # joint6: wrist yaw
            ]
        )
        # Simulate for some time to let the system stabilize
        self.reset(True)
        mujoco.mj_step(self.model, self.data)
        if self.render:
            self.viewer.sync()
        self.nv = self.model.nv
        self.jacp = np.zeros((3, self.nv))
        self.jacr = np.zeros((3, self.nv))
        self.M = np.zeros((self.nv, self.nv))
        self.latest_command_stamp = time.time()
        self.actuator_tau = np.zeros(6)
        self.tau_ff = np.zeros(6)
        self.dq_des = np.zeros(6)
        # Fix URDF and meshes path
        urdf_path = os.path.join(SCRIPT_DIR, "z1.urdf")
        if not os.path.exists(urdf_path):
            raise FileNotFoundError(f"z1.urdf not found at {urdf_path}")
        # Use assets directory for meshes
        meshes_dir = ASSETS_PATH
        if not os.path.exists(meshes_dir):
            raise FileNotFoundError(f"Assets directory not found at {meshes_dir}")
        # Debug: List available mesh files
        print(f"Looking for meshes in: {meshes_dir}")
        if os.path.exists(meshes_dir):
            mesh_files = [f for f in os.listdir(meshes_dir) if f.endswith(".stl")]
            print(f"Found STL mesh files: {mesh_files}")
        try:
            # Method 1: Try with mesh directory
            print("Attempting to load Pinocchio model with mesh directory...")
            self.pin_model = RobotWrapper.BuildFromURDF(urdf_path, meshes_dir)
            print("✓ Pinocchio model loaded successfully with meshes!")
        except Exception as e1:
            print(f"✗ Method 1 failed: {e1}")
            try:
                # Method 2: Try with list of directories
                print("Attempting to load with directory list...")
                self.pin_model = RobotWrapper.BuildFromURDF(urdf_path, [meshes_dir])
                print("✓ Pinocchio model loaded successfully with directory list!")
            except Exception as e2:
                print(f"✗ Method 2 failed: {e2}")
                try:
                    # Method 3: Load without meshes (visual only)
                    print("Attempting to load without meshes...")
                    self.pin_model = RobotWrapper.BuildFromURDF(urdf_path)
                    print("✓ Pinocchio model loaded without meshes (kinematics only)!")
                except Exception as e3:
                    print(f"✗ Method 3 failed: {e3}")
                    print(
                        "⚠️  Could not load Pinocchio model - kinematics functions will not work"
                    )
                    self.pin_model = None
        # ROS 2 setup
        if not rclpy.ok():
            rclpy.init()
        self.ros_node = rclpy.create_node("z1sim_torque_sender")
        self.torque_pub = self.ros_node.create_publisher(
            Float64MultiArray,
            "/z1_controller/commands",
            10,
        )
        # Joint state subscription
        self.joint_state_sub = self.ros_node.create_subscription(
            JointState, "/joint_states", self.joint_state_callback, 10
        )
        self.latest_joint_states = {}
        time.sleep(1)  # Allow time for the initial state to be set

    def forward_kinematics(self, q, update=True):
        """
        Compute the forward kinematics for the end-effector frame.
        Args:
            q (np.ndarray): Joint positions (size: [model.nq] or [only joint DOFs])
        Returns:
            T_S_F (pinocchio.SE3): Transformation matrix of the end-effector frame.
        """
        if self.pin_model is None:
            raise RuntimeError("Pinocchio model not available")
        T_S_F = self.pin_model.framePlacement(
            q, END_EFF_FRAME_ID, update_kinematics=update
        )
        return np.array(T_S_F)

    def joint_state_callback(self, msg):
        """
        Callback to handle incoming joint states.
        """
        positions = list(msg.position[:6])
        velocities = list(msg.velocity[:6])
        efforts = list(msg.effort[:6])
        names = list(msg.name[:6])  # Ensure we only take the first 6 names
        self.latest_joint_states = {
            "names": names,
            "positions": positions,
            "velocities": velocities,
            "efforts": efforts,
        }

    def get_latest_joint_states(self):
        """
        Returns the most recent joint states received from the /joint_states topic.
        Returns:
            dict: {'names', 'positions', 'velocities', 'efforts'}
        """
        return self.latest_joint_states

    def send_torque_to_ros_node(self, torque):
        """
        Publishes a 6-DOF torque vector to the ROS 2 topic expected by the Z1 controller.
        """
        if len(torque) != 6:
            raise ValueError("Torque command must be a 6-element vector.")
        msg = Float64MultiArray()
        msg.data = torque.tolist()
        self.torque_pub.publish(msg)
        rclpy.spin_once(self.ros_node, timeout_sec=0.001)

    def reset(self, active=True):
        self.data.qpos[:6] = self.q0
        self.data.qvel[:6] = np.zeros(6)
        mujoco.mj_step(self.model, self.data)
        if self.render and (self.step_counter % self.render_ds_ratio) == 0:
            self.viewer.sync()

    def get_state(self):
        return self.data.qpos[:6], self.data.qvel[:6]

    def get_joint_acceleration(self):
        return self.data.qacc[:6]

    def send_joint_torque(self, torques):
        self.tau_ff = torques
        self.latest_command_stamp = time.time()
        self.step()

    def step(self):
        tau = self.tau_ff
        self.actuator_tau = tau
        self.data.ctrl[:6] = tau.squeeze()
        self.step_counter += 1
        mujoco.mj_step(self.model, self.data)
        if self.render and (self.step_counter % self.render_ds_ratio) == 0:
            self.viewer.sync()

    def get_gravity(self, q):
        if self.pin_model is None:
            raise RuntimeError("Pinocchio model not available")
        g = self.pin_model.gravity(q)
        return g[:6]

    def get_dynamics(self, q, v):
        """
        Compute the joint-space inertia matrix M(q) and the nonlinear effects h(q, v) = C(q, v)*v + g(q).
        Args:
            q (np.ndarray): Joint positions (size: [model.nq] or [only joint DOFs])
            v (np.ndarray): Joint velocities (size: [model.nv] or [only joint DOFs])
        Returns:
            M (np.ndarray): Mass matrix (nv x nv)
            h (np.ndarray): Nonlinear effects (nv,)
        """
        if self.pin_model is None:
            raise RuntimeError("Pinocchio model not available")
        M = self.pin_model.mass(q)
        h = self.pin_model.nle(q, v)
        return M[:6, :6], h[:6]

    def get_site_pose(self, site_name):
        """Get the world position of a specific site."""
        return self.data.site(site_name).xpos

    def get_site_id(self, site_name):
        """
        Get the ID of a specific site by its name.
        Args:
            site_name (str): Name of the site to get the ID for.
        Returns:
            int: Site ID.
        """
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)

    def get_body_id(self, body_name):
        """
        Get the ID of a specific body by its name.
        Args:
            body_name (str): Name of the body to get the ID for.
        Returns:
            int: Body ID.
        """
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)

    def get_body_names(self):
        """Get a list of all body names in the model.
        Returns:
            list: List of body names.
        """
        return [self.model.body(i).name for i in range(self.model.nbody)]

    def get_body_mass(self, body_name):
        """
        Get the mass of a specific body by its name.
        Args:
            body_name (str): Name of the body to get the mass for.
        Returns:
            float: Mass of the body
        """
        body_id = self.get_body_id(body_name)
        return self.model.body_mass[body_id]

    def get_body_pose(self, body_name):
        """
        Get the pose of a specific body in the world frame.
        Args:
            body_name (str): Name of the body to get the pose for.
        Returns:
            T (np.ndarray): 4x4 transformation matrix representing the pose.
        """
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        T = self.data.body(body_id).xpos
        quat = self.data.body(body_id).xquat
        return (
            np.array(pin.SE3(R.from_quat(quat).as_matrix(), T).homogeneous)[:3, :3],
            np.array(pin.SE3(R.from_quat(quat).as_matrix(), T).homogeneous)[:3, 3],
        )

    def compute_tau(self, q, v, ddq_des):
        """
        Compute the total torque command using inverse dynamics:
        tau = M(q) * ddq_des + h(q, v)
        Args:
            q (np.ndarray): Joint positions
            v (np.ndarray): Joint velocities
            ddq_des (np.ndarray): Desired joint accelerations
        Returns:
            tau (np.ndarray): Joint torques
        """
        M, h = self.get_dynamics(q, v)
        tau = M @ ddq_des + h
        return tau[:6]

    def get_gravity_vector(self):
        """
        Get the gravity vector in the world frame.
        Returns:
            np.ndarray: Gravity vector (3,)
        """
        return self.model.opt.gravity

    def get_jacobian(self, q):
        if self.pin_model is None:
            raise RuntimeError("Pinocchio model not available")
        J_temp = self.pin_model.computeFrameJacobian(q, END_EFF_FRAME_ID)
        J = np.zeros([6, 6])
        J[3:6, :] = J_temp[0:3, :6]
        J[0:3, :] = J_temp[3:6, :6]
        return J

    def get_pose(self, q):
        if self.pin_model is None:
            raise RuntimeError("Pinocchio model not available")
        T_S_F = self.pin_model.framePlacement(q, END_EFF_FRAME_ID)
        return T_S_F.homogeneous

    def get_ee_force_torque(self):
        try:
            force_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SENSOR, "ee_force"
            )
            torque_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SENSOR, "ee_torque"
            )
            force = self.data.sensordata[force_id : force_id + 3]
            torque = self.data.sensordata[torque_id : torque_id + 3]
            return force.copy(), torque.copy()
        except:
            # Return zeros if sensors are not available
            return np.zeros(3), np.zeros(3)

    def close(self):
        if self.render:
            self.viewer.close()
        self.ros_node.destroy_node()
        rclpy.shutdown()

    @staticmethod
    def slerp_trajectory(start_pos, end_pos, start_quat, end_quat, steps):
        start_quat = np.asarray(start_quat) / np.linalg.norm(start_quat)
        end_quat = np.asarray(end_quat) / np.linalg.norm(end_quat)
        positions = np.linspace(start_pos, end_pos, steps)
        r_start = R.from_quat(start_quat)
        r_end = R.from_quat(end_quat)
        key_rots = R.concatenate([r_start, r_end])
        slerp = R.slerp(0, 1, key_rots)
        interp_rots = slerp(np.linspace(0, 1, steps))
        traj = np.zeros((steps, 4, 4))
        for i in range(steps):
            traj[i, :3, :3] = interp_rots[i].as_matrix()
            traj[i, :3, 3] = positions[i]
            traj[i, 3, :] = [0, 0, 0, 1]
        return traj

    @staticmethod
    def linear_trajectory(start_pos, end_pos, steps):
        return np.linspace(start_pos, end_pos, steps)

    def slerp_orientation(self, current_quat, target_quat, steps):
        current_quat = np.asarray(current_quat) / np.linalg.norm(current_quat)
        target_quat = np.asarray(target_quat) / np.linalg.norm(target_quat)
        r_start = R.from_quat(current_quat)
        r_end = R.from_quat(target_quat)
        key_rots = R.concatenate([r_start, r_end])
        slerp = R.slerp(0, 1, key_rots)
        interp_rots = slerp(np.linspace(0, 1, steps))
        return interp_rots.as_quat()
