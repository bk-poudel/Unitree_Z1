#!/usr/bin/env python3
import os
import re

def fix_urdf_for_local_meshes():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    urdf_path = os.path.join(script_dir, "z1.urdf")
    
    if not os.path.exists(urdf_path):
        print(f"URDF file not found: {urdf_path}")
        return
    
    # Create backup
    backup_path = urdf_path + '.backup'
    
    # Read URDF content
    with open(urdf_path, 'r') as f:
        content = f.read()
    
    # Save backup
    with open(backup_path, 'w') as f:
        f.write(content)
    print(f"Backup created: {backup_path}")
    
    # Fix mesh paths to point to local z1_description directory
    # Replace package:// references with local file paths
    
    # Pattern 1: Visual meshes (DAE files)
    content = re.sub(
        r'filename="package://z1_description/meshes/visual/([^"]+)"',
        r'filename="z1_description/meshes/visual/\1"',
        content
    )
    
    # Pattern 2: Collision meshes (STL files)  
    content = re.sub(
        r'filename="package://z1_description/meshes/collision/([^"]+)"',
        r'filename="z1_description/meshes/collision/\1"',
        content
    )
    
    # Pattern 3: Any other package references
    content = re.sub(
        r'filename="package://z1_description/([^"]+)"',
        r'filename="z1_description/\1"',
        content
    )
    
    # Write the fixed URDF
    with open(urdf_path, 'w') as f:
        f.write(content)
    
    print("URDF mesh paths fixed to use local files!")
    
    # Show the changes
    print("\nUpdated mesh references:")
    mesh_lines = [line.strip() for line in content.split('\n') if 'mesh' in line and 'filename' in line]
    for line in mesh_lines[:5]:  # Show first 5
        print(f"  {line}")
    if len(mesh_lines) > 5:
        print(f"  ... and {len(mesh_lines) - 5} more")

if __name__ == "__main__":
    fix_urdf_for_local_meshes()