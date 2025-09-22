# This script demonstrates how to simulate the movement of a robot leg using PyBullet.
# It requires a URDF (Unified Robot Description Format) file for the leg model.
#
# A sample URDF is provided below as a multi-line string.
# You can save this content as 'simple_leg.urdf' and place it in the same directory as this script.

import pybullet as p
import pybullet_data
import time
import math

# --- Start of the main script ---

def run_simulation():
    """
    Main function to set up and run the PyBullet simulation.
    """
    # 1. Connect to the PyBullet physics server in GUI mode for visualization.
    #    You can use p.DIRECT for non-graphical simulations (e.g., for faster training).
    print("Connecting to PyBullet physics client...")
    physics_client = p.connect(p.GUI)

    # 2. Set up the simulation environment.
    p.setAdditionalSearchPath(pybullet_data.getDataPath())  # Used for finding default URDF files
    p.setGravity(0, 0, -9.81)  # Set gravity along the z-axis
    p.loadURDF("plane.urdf")   # Load a simple ground plane

    # 3. Load the robot leg model.
    # We've added a starting position [x, y, z] to place the leg higher in the air.
    # Replace "simple_leg.urdf" with the path to your robot's URDF file.
    # 'useFixedBase=True' pins the leg's base, simulating it being attached to a body.
    try:
        start_position = [0, 0, 1]  # Set the initial position 0.5 meters high
        robot_leg_id = p.loadURDF("simple_leg.urdf", start_position, useFixedBase=False)
    except p.error as e:
        print(f"Error loading URDF file: {e}")
        print("Please ensure 'simple_leg.urdf' exists in the same directory or provide the correct path.")
        p.disconnect()
        return

    # 4. Get information about the robot's joints.
    num_joints = p.getNumJoints(robot_leg_id)
    print(f"Loaded a robot with {num_joints} joints.")
    
    # Let's find a revolute joint to control.
    target_joint_index = -1
    for i in range(num_joints):
        joint_info = p.getJointInfo(robot_leg_id, i)
        joint_name = joint_info[1].decode("utf-8")
        joint_type = joint_info[2]
        print(f"Joint {i}: Name='{joint_name}', Type='{joint_type}'")
        
        # Check if the joint is a revolute (rotational) joint.
        if joint_type == p.JOINT_REVOLUTE:
            target_joint_index = i
            print(f"Found a revolute joint at index {target_joint_index}. Will control this joint.")
            break
    
    if target_joint_index == -1:
        print("No revolute joint found in the URDF. The script will not control any joint.")
        p.disconnect()
        return

    # 5. Control the target joint in a continuous loop.
    print("Starting simulation loop. Press Ctrl+C in the terminal to exit.")
    start_time = time.time()
    try:
        while True:
            # Use a sine wave to create a smooth back-and-forth motion.
            # The angle will oscillate between -pi/2 and +pi/2 radians (-90 to +90 degrees).
            elapsed_time = time.time() - start_time
            target_angle = math.sin(elapsed_time * 2) * (math.pi / 2)

            # Set the joint motor control. POSITION_CONTROL mode will command the joint to
            # move to the specified 'targetPosition'.
            p.setJointMotorControl2(
                bodyUniqueId=robot_leg_id,
                jointIndex=target_joint_index,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target_angle,
                force=50  # Apply a force to ensure it reaches the target
            )

            # Step the simulation forward.
            p.stepSimulation()

            # Add a small delay to make the simulation visible.
            # 1/240.0 is the default simulation time step (240 Hz).
            time.sleep(1.0 / 240.0)

    except KeyboardInterrupt:
        # Gracefully handle the exit when the user presses Ctrl+C.
        print("\nSimulation stopped by user.")
    
    finally:
        # Disconnect from the physics server.
        p.disconnect()
        print("PyBullet client disconnected.")

# The sample URDF for a simple two-link leg.
# Save this content as a file named 'simple_leg.urdf'.
# This model has two links and a single revolute joint to demonstrate movement.
sample_urdf = """<?xml version="1.0"?>
<robot name="simple_leg">
    <link name="base_link">
        <visual>
            <geometry>
                <box size="0.1 0.1 0.1"/>
            </geometry>
            <material name="white">
                <color rgba="1 1 1 1"/>
            </material>
        </visual>
        <!-- Added a collision shape for the base link. -->
        <collision>
            <geometry>
                <box size="0.1 0.1 0.1"/>
            </geometry>
        </collision>
    </link>

    <joint name="hip_joint" type="revolute">
        <parent link="base_link"/>
        <child link="thigh_link"/>
        <axis xyz="1 0 0"/>
        <limit lower="-1.57" upper="1.57" effort="50" velocity="10"/>
        <origin rpy="0 0 0" xyz="0 0 0"/>
    </joint>

    <link name="thigh_link">
        <visual>
            <origin rpy="0 0 0" xyz="0 0 -0.25"/>
            <geometry>
                <box size="0.05 0.05 0.5"/>
            </geometry>
            <material name="blue">
                <color rgba="0 0 1 1"/>
            </material>
        </visual>
        <!-- Added a collision shape for the thigh link. -->
        <collision>
            <origin rpy="0 0 0" xyz="0 0 -0.25"/>
            <geometry>
                <box size="0.05 0.05 0.5"/>
            </geometry>
        </collision>
        <inertial>
            <mass value="1"/>
            <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
        </inertial>
    </link>
</robot>
"""

# Create the URDF file from the string content.
# This makes the script self-contained and ready to run.
if __name__ == "__main__":
    with open("simple_leg.urdf", "w") as f:
        f.write(sample_urdf)
    print("Created 'simple_leg.urdf' file.")
    run_simulation()
