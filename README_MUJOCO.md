# MuJoCo Scene Loader

This directory contains simple Python scripts to load and run the Unitree Z1 robot simulation using MuJoCo.

## Files

- `scene.xml` - Main MuJoCo scene file (includes the Z1 robot)
- `z1.xml` - Z1 robot model definition
- `simple_mujoco.py` - **Recommended** - Minimal script to load and run the scene
- `requirements.txt` - Python dependencies

## Installation

1. Install MuJoCo Python bindings:
```bash
pip install mujoco
```

2. Or install all requirements:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage (No GUI)
```bash
python simple_mujoco.py
```

This will:
- Load the scene.xml file
- Print model information
- Run a basic simulation
- Attempt to open a 3D viewer (if available)

### What the scene contains

The `scene.xml` file includes:
- Unitree Z1 robotic arm (loaded from `z1.xml`)
- Ground plane with checker texture
- Proper lighting and visual settings
- Skybox background

## Expected Output

When you run the script, you should see:
```
Loading scene.xml...
Model loaded successfully!
Bodies: X, Joints: Y, Actuators: Z
Running simulation...
Step 0: t=0.00s
Step 25: t=0.50s
...
Opening viewer (press ESC to close)...
```

## Troubleshooting

### "MuJoCo not installed"
Run: `pip install mujoco`

### "scene.xml not found"
Make sure you're running the script from the same directory as `scene.xml`

### "Viewer not available"
The basic simulation will still work. For full viewer support, ensure you have:
- OpenGL support
- Display available (not running headless)

### Model loading errors
Check that:
- `z1.xml` exists in the same directory
- All `.stl` files in the `assets/` folder are present
- XML files have valid syntax

## Advanced Usage

The script tries to:
1. Load the model from `scene.xml`
2. Create simulation data
3. Run basic physics simulation
4. Open interactive 3D viewer

You can modify `simple_mujoco.py` to:
- Add robot control
- Change simulation parameters  
- Record data or videos
- Add custom sensors

## MuJoCo Documentation

- [MuJoCo Python Documentation](https://mujoco.readthedocs.io/en/latest/python.html)
- [MuJoCo Modeling Guide](https://mujoco.readthedocs.io/en/latest/modeling.html)
