# This script demonstrates how to simulate the movement of a robot leg using PyBullet.
# It requires a URDF (Unified Robot Description Format) file for the leg model.

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
    # The URDF file for the quadruped is now named 'simple_quadruped.urdf'.
    try:
        # Set the starting position to be 1.0 meter high.
        start_position = [0, 0, 1.0]
        # useFixedBase=True to start, but you can set to False to test the full dynamic model
        robot_id = p.loadURDF("simple_quadruped.urdf", start_position, useFixedBase=False)
    except p.error as e:
        print(f"Error loading URDF file: {e}")
        print("Please ensure 'simple_quadruped.urdf' exists in the same directory or provide the correct path.")
        p.disconnect()
        return

    # 4. Get information about the robot's joints.
    num_joints = p.getNumJoints(robot_id)
    print(f"Loaded a robot with {num_joints} joints.")
    
    # Let's find all the revolute joints to control.
    joint_indices = []
    for i in range(num_joints):
        joint_info = p.getJointInfo(robot_id, i)
        joint_name = joint_info[1].decode("utf-8")
        joint_type = joint_info[2]
        print(f"Joint {i}: Name='{joint_name}', Type='{joint_type}'")
        
        if joint_type == p.JOINT_REVOLUTE:
            joint_indices.append(i)
    
    if not joint_indices:
        print("No revolute joints found in the URDF. The script will not control any joint.")
        p.disconnect()
        return

    # 5. Control the target joints in a continuous loop.
    print(f"Found {len(joint_indices)} revolute joints. Starting simulation loop. Press Ctrl+C in the terminal to exit.")
    start_time = time.time()
    try:
        while True:
            # Use a sine wave to create a smooth back-and-forth motion.
            # The angle will oscillate between -pi/2 and +pi/2 radians (-90 to +90 degrees).
            elapsed_time = time.time() - start_time
            target_angle = math.sin(elapsed_time * 2) * (math.pi / 2)

            # Control all found revolute joints with the same target angle.
            for joint_index in joint_indices:
                p.setJointMotorControl2(
                    bodyUniqueId=robot_id,
                    jointIndex=joint_index,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=target_angle,
                    force=20
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

if __name__ == "__main__":
    run_simulation()
