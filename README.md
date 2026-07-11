# Unitree Z1 MuJoCo Simulation
**Project:** Unitree Z1 robot simulation and control demos  
**Maintainer:** Bibek Poudel <bp2376@nyu.edu>
<p float="left">
  <img src="z1.png" width="420" alt="Unitree Z1 robot arm">
</p>
## Overview
This repository contains a MuJoCo-based simulation setup for the Unitree Z1 robotic arm. It includes the robot model, scene definitions, and several Python scripts for testing motion, inverse kinematics, torque control, pick-and-place behavior, and visualization.
The project is organized around a few core assets:
- `scene.xml` and `z1.xml` for the MuJoCo scene and robot model
- `z1.urdf` and `z1_description/` for the URDF and mesh assets
- Python demos such as `simple.py`, `Z1Sim.py`, `Force_control.py`, and the pick-and-place examples
## What You Can Do Here
- Load and inspect the Unitree Z1 model in MuJoCo
- Run passive viewer demos and basic motion tests
- Experiment with torque and position control
- Try inverse kinematics and trajectory interpolation
- Prototype pick-and-place and manipulation workflows
## Requirements
- Python 3.9+ recommended
- MuJoCo `3.x`
- NumPy
- Pinocchio (`pin` or `pinocchio`, depending on your environment)
The pinned Python dependencies are listed in [requirements.txt](requirements.txt).
## Installation
Install the Python packages manually:
```bash
pip install -r requirements.txt
```
Or use the helper script:
```bash
bash install_dependencies.sh
```
If `pin` fails to install in your environment, install Pinocchio through your platform package manager or conda-forge.
## Quick Start
Start with the simplest demo first:
```bash
python simple.py
```
Other useful entry points:
- `python Z1Sim.py` for the combined MuJoCo + Pinocchio simulation wrapper
- `python Force_control.py` for torque-focused tests
- `python visualization.py` for a lightweight viewer example
## Script Guide
| Script | Purpose |
| --- | --- |
| `simple.py` | Loads `scene.xml`, prints model information, and runs basic joint and torque tests |
| `simple_working.py` | Minimal working MuJoCo scene example |
| `Z1Sim.py` | Simulation wrapper combining MuJoCo with Pinocchio-based kinematics/dynamics helpers |
| `Force_control.py` | Force and torque control experiments |
| `pick_and_place_initial.py` / `pick_and_place_final2.py` | Pick-and-place and manipulation experiments |
| `pick_place_interpolated.py` | Trajectory interpolation between target poses |
| `moving_forward_inverse_kinematics.py` | Forward motion and IK-based motion tests |
| `visualization.py` | Viewer-oriented scene loading example |
| `fix_urdf_local_paths.py` | Utility for adjusting URDF mesh paths for the local checkout |
## Notes On Paths
Some of the older demo scripts were written with hardcoded mesh paths. If a script fails to find meshes or URDF assets, run `fix_urdf_local_paths.py` or update the asset paths in the script to point at your local checkout.
## Repository Layout
- `scene.xml` - Main MuJoCo scene file
- `z1.xml` - Robot model definition used by the scene
- `z1.urdf` - URDF version of the Z1 arm
- `assets/` - Example object assets used for manipulation scenes
- `z1_description/` - Meshes, xacro files, and ROS-style robot description files
- `examples_py/` - Additional example code and interface files
## Troubleshooting
- If MuJoCo cannot load the model, confirm that you are running commands from the repository root.
- If the viewer does not open, check your OpenGL/display setup.
- If a pick-and-place script fails on mesh loading, check the URDF asset paths first.
- If you are using a new Python environment, reinstall the dependencies after activation.
## License
This project is released under a [BSD-3-Clause License](LICENSE).
