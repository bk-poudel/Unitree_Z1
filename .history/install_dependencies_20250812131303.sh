#!/bin/bash
# Installation script for Z1 Arm Controller dependencies

echo "Installing Z1 Arm Controller dependencies..."

# Update pip
python3 -m pip install --upgrade pip

# Install basic requirements
python3 -m pip install numpy>=1.20.0

# Install MuJoCo
echo "Installing MuJoCo..."
python3 -m pip install mujoco>=3.0.0

# Install Pinocchio (this might take some time)
echo "Installing Pinocchio..."
python3 -m pip install pin

# Alternative Pinocchio installation methods if the above fails:
# For Ubuntu/Debian:
# sudo apt update
# sudo apt install robotpkg-py3*-pinocchio

# For conda users:
# conda install pinocchio -c conda-forge

echo "Installation complete!"
echo "You can now run the Z1 arm controller with: python3 z1_arm_controller_fixed.py"
