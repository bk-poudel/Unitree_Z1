import mujoco
import pinocchio as pin
import numpy as np
import time
import mujoco.viewer


class Z1Sim:
    def __init__(self, mjcf_path, urdf_path):
        # Load Mujoco model
        self.model = mujoco.MjModel.from_xml_path(mjcf_path)
        self.data = mujoco.MjData(self.model)
        # Load Pinocchio model
        self.pin_model = pin.buildModelFromUrdf(urdf_path)
        self.pin_data = self.pin_model.createData()
        # Simulation parameters
        self.dt = self.model.opt.timestep
        # Controller gains for position control
        self.kp = 50.0  # Proportional gain
        self.kd = 5.0   # Derivative gain
        # Number of joints (assuming 6-DOF arm)
        self.num_joints = 6

    def disable_dynamics(self):
        """Disable most dynamics effects for simplified simulation"""
        # Disable gravity
        self.model.opt.gravity[2] = 0.0
        
        # Disable joint damping
        self.model.dof_damping[:] = 0.0
            
        # Disable friction
        for i in range(self.model.ngeom):
            self.model.geom_friction[i][:] = 0.0
            
        # Disable armature (rotor inertia)
        self.model.dof_armature[:] = 0.0
        
        # Disable contact
        self.model.opt.disableflags = mujoco.mjtDisableBit.mjDSBL_CONTACT

    def reset(self, q=None, qdot=None):
        # Reset Mujoco state
        if q is not None:
            self.data.qpos[:] = q
        else:
            self.data.qpos[:] = 0
        if qdot is not None:
            self.data.qvel[:] = qdot
        else:
            self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)

    def step(self, ctrl):
        # Apply control and step simulation
        self.data.ctrl[:] = ctrl
        mujoco.mj_step(self.model, self.data)

    def get_state(self):
        # Return current state
        return np.copy(self.data.qpos), np.copy(self.data.qvel)

    def set_state(self, q, qdot):
        self.data.qpos[:] = q
        self.data.qvel[:] = qdot
        mujoco.mj_forward(self.model, self.data)

    def compute_pinocchio_dynamics(self, q, qdot, tau):
        # Use Pinocchio for forward dynamics
        pin.forwardDynamics(self.pin_model, self.pin_data, q, qdot, tau)
        return self.pin_data.ddq

    def compute_pinocchio_kinematics(self, q):
        # Use Pinocchio for forward kinematics
        pin.forwardKinematics(self.pin_model, self.pin_data, q)
        pin.updateFramePlacements(self.pin_model, self.pin_data)
        return self.pin_data.oMf

    def render(self):
        # Simple viewer
        mujoco.viewer.launch_passive(self.model, self.data)

    def send_torque(self, tau):
        # Send torque to actuators
        self.data.ctrl[:self.num_joints] = tau
        mujoco.mj_step(self.model, self.data)
    
    def position_control(self, target_positions):
        """
        Implements PD position control for the robot joints
        
        Args:
            target_positions: Array of target joint positions (radians)
        """
        # Get current state
        q, qdot = self.get_state()
        
        # Calculate position error
        pos_error = target_positions - q[:self.num_joints]
        
        # Calculate velocity error (target velocity is 0 for position control)
        vel_error = 0 - qdot[:self.num_joints]
        
        # PD control: τ = Kp * (qdes - q) + Kd * (qdot_des - qdot)
        torque = self.kp * pos_error + self.kd * vel_error
        
        # Apply torque command
        self.send_torque(torque)
        
        return pos_error


# Example usage:
if __name__ == "__main__":
    mjcf_path = "scene.xml"
    urdf_path = "z1.urdf"
    sim = Z1Sim(mjcf_path, urdf_path)
    sim.reset()
    
    # Create viewer
    viewer = mujoco.viewer.launch_passive(sim.model, sim.data)
    
    # Target positions (in radians) for each joint
    target_positions = np.array([0.5, 0.3, 0.2, 0.1, -0.2, 0.0])
    
    # Run position control loop
    start_time = time.time()
    while time.time() - start_time < 10:
        # Apply position control
        error = sim.position_control(target_positions)
        
        # Synchronize the viewer
        viewer.sync()
        
        # Print position error norm (optional)
        if int(time.time() * 10) % 10 == 0:  # print every ~0.1 seconds
            print(f"Position error: {np.linalg.norm(error):.4f}")
            
        time.sleep(0.01)  # Small delay for stability
    
    # Close viewer
    viewer.close()
