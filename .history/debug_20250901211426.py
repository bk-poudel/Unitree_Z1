import mujoco

# Load the model from the XML file
try:
    model = mujoco.MjModel.from_xml_path("./scene.xml")
    data = mujoco.MjData(model)
except FileNotFoundError:
    print(
        "Error: 'z1.xml' not found. Please ensure the file is in the correct directory."
    )
    exit()
print("--- All Body Names ---")
# The loop should find all bodies, including the nested ones
for i in range(model.nbody):
    body_name = model.body(i).name
    print(f"Body {i}: {body_name}")
print("\n--- Check for gripperMover ---")
# Explicitly check for the 'gripperMover' body by name to confirm it exists
try:
    gripper_mover_id = model.body("gripperMover").id
    print(f"Success: 'gripperMover' body found with ID: {gripper_mover_id}")
except KeyError:
    print("Error: 'gripperMover' body was not found.")
