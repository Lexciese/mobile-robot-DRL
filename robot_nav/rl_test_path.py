from models.TD3.TD3 import TD3

import torch
import numpy as np
from robot_nav.SIM_ENV.sim import SIM
import yaml

# Add ROS 2 imports
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
import math


def main(args=None):
    """Main testing function"""
    action_dim = 2  # number of actions produced by the model
    max_action = 1  # maximum absolute value of output actions
    state_dim = 12  # number of input values in the neural network (vector length of state input)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # using cuda if it is available, cpu otherwise
    epoch = 0  # epoch number
    max_steps = 300  # maximum number of steps in single episode

    # Configuration options
    use_real_robot = True  # Set to True for real robot testing, False for pure simulation
    real_robot_timeout = 600  # Much longer timeout for real robot (in seconds)
    goal_distance_threshold = 0.2  # Distance in meters to consider the goal reached
    collision_distance_threshold = 0.3  # Distance in meters to consider a collision
    status_update_interval = 20  # Print status update every N iterations
    ignore_collisions = True  # When True, robot will continue execution even when collision is detected
    
    # Velocity scaling for real robot (adjust these based on real robot performance)
    linear_velocity_scale = 1.0  # Scale factor for linear velocity
    angular_velocity_scale = 3.0  # Scale factor for angular velocity
    max_linear_velocity =,2.0  # Maximum linear velocity in m/s
    max_angular_velocity = 4.0  # Maximum angular velocity in rad/s
    
    # Initialize ROS 2
    rclpy.init(args=args)
    ros_node = Node('rl_test_path_node')
    
    # Create publisher for cmd_vel
    cmd_vel_publisher = ros_node.create_publisher(Twist, '/cmd_vel', 10)
    twist_msg = Twist()
    
    # Create variables to store the latest scan data from ROS
    latest_ros_scan = None
    ros_scan_received = False
    
    # Create variables to store the latest odometry data from ROS
    latest_ros_odom = None
    ros_odom_received = False
    
    # Callback for laser scan subscriber
    def scan_callback(msg):
        nonlocal latest_ros_scan, ros_scan_received
        latest_ros_scan = msg.ranges
        ros_scan_received = True
    
    # Callback for odometry subscriber
    def odom_callback(msg):
        nonlocal latest_ros_odom, ros_odom_received
        latest_ros_odom = msg
        ros_odom_received = True
    
    # Create subscriber for scan
    scan_subscriber = ros_node.create_subscription(
        LaserScan,
        '/scan',
        scan_callback,
        10)
        
    # Create subscriber for odometry
    odom_subscriber = ros_node.create_subscription(
        Odometry,
        '/odom',
        odom_callback,
        10)

    model = TD3(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        device=device,
        load_model=True,
        model_name="TD3",
    )  # instantiate a model

    sim = SIM(world_file="worlds/empty.yaml")  # instantiate environment
    with open("robot_nav/eval_points.yaml") as file:
        points = yaml.safe_load(file)
    robot_poses = points["robot"]["poses"]
    robot_goals = points["robot"]["goals"]

    print("..............................................")
    print(f"Following path with {len(robot_goals)} waypoints...")
    total_reward = 0.0
    total_steps = 0
    col = 0
    goals = 0

    # Start at (0,0,0)
    robot_state = [[0], [0], [0]]
    latest_scan, distance, cos, sin, collision, goal, a, reward = sim.reset(
        robot_state=robot_state,
        robot_goal=robot_goals[0],
        random_obstacles=False,
    )

    try:
        for idx in range(len(robot_goals)):
            # Fix: Extract scalar values from goal data, handling nested lists
            goal_data = robot_goals[idx]
            
            # Debug the structure of goal data
            print(f"Debug - Goal data format: {goal_data}")
            
            # Extract x coordinate as scalar value
            if isinstance(goal_data, (list, tuple)) and len(goal_data) > 0:
                goal_x = float(goal_data[0]) if not isinstance(goal_data[0], (list, tuple)) else float(goal_data[0][0])
            else:
                goal_x = 0.0
                
            # Extract y coordinate as scalar value
            if isinstance(goal_data, (list, tuple)) and len(goal_data) > 1:
                goal_y = float(goal_data[1]) if not isinstance(goal_data[1], (list, tuple)) else float(goal_data[1][0])
            else:
                goal_y = 0.0
            
            print(f"\nMoving to goal {idx+1}/{len(robot_goals)}: ({goal_x:.2f}, {goal_y:.2f})")
            sim.set_robot_goal(robot_goals[idx])
            count = 0
            done = False
            goal_start_time = ros_node.get_clock().now()
            
            # Reset for new goal
            distance_to_goal = float('inf')
            
            # Use different max steps based on mode
            current_max_steps = real_robot_timeout if use_real_robot else max_steps
            
            while not done and count < current_max_steps:
                # Process any pending ROS callbacks
                rclpy.spin_once(ros_node, timeout_sec=0)
                
                # Status update at intervals
                if count % status_update_interval == 0:
                    if ros_odom_received:
                        elapsed_time = (ros_node.get_clock().now() - goal_start_time).nanoseconds / 1e9
                        print(f"[{elapsed_time:.1f}s] Current pos: ({latest_ros_odom.pose.pose.position.x:.2f}, " +
                              f"{latest_ros_odom.pose.pose.position.y:.2f}), Distance to goal: {distance_to_goal:.2f}m")
                
                # Use ROS scan data if available, otherwise use simulation data
                scan_data = latest_ros_scan if ros_scan_received else latest_scan
                
                # Always prioritize real odometry data when available
                if ros_odom_received:
                    # Extract position from odometry
                    current_x = latest_ros_odom.pose.pose.position.x
                    current_y = latest_ros_odom.pose.pose.position.y
                    
                    # Extract orientation from odometry (quaternion)
                    qx = latest_ros_odom.pose.pose.orientation.x
                    qy = latest_ros_odom.pose.pose.orientation.y
                    qz = latest_ros_odom.pose.pose.orientation.z
                    qw = latest_ros_odom.pose.pose.orientation.w
                    
                    # Calculate yaw from quaternion
                    siny_cosp = 2 * (qw * qz + qx * qy)
                    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
                    current_yaw = math.atan2(siny_cosp, cosy_cosp)
                    
                    # Calculate distance to goal
                    dx = goal_x - current_x
                    dy = goal_y - current_y
                    distance_to_goal = math.sqrt(dx * dx + dy * dy)
                    
                    # Calculate angle to goal
                    angle_to_goal = math.atan2(dy, dx)
                    angle_diff = angle_to_goal - current_yaw
                    
                    # Normalize angle
                    while angle_diff > math.pi:
                        angle_diff -= 2 * math.pi
                    while angle_diff < -math.pi:
                        angle_diff += 2 * math.pi
                    
                    # Set values for the agent
                    distance = distance_to_goal
                    cos_angle = math.cos(angle_diff)
                    sin_angle = math.sin(angle_diff)
                    
                    # Update goal status based on real position
                    goal = distance_to_goal < goal_distance_threshold
                    
                    # Update collision status if scan data available
                    if ros_scan_received and len(latest_ros_scan) > 0:
                        try:
                            # Add safety check for non-positive values
                            valid_ranges = [x for x in latest_ros_scan if x > 0]
                            min_scan_distance = min(valid_ranges) if valid_ranges else float('inf')
                            collision = min_scan_distance < collision_distance_threshold
                        except Exception as e:
                            print(f"Error processing scan data: {e}")
                            collision = False
                else:
                    # Use simulation data if no odometry available
                    distance = distance
                    cos_angle = cos
                    sin_angle = sin
                
                state, terminal = model.prepare_state(
                    scan_data, distance, cos_angle, sin_angle, collision, goal, a
                )
                action = model.get_action(np.array(state), False)
                a_in = [(action[0] + 1) / 4, action[1]]
                
                # Apply velocity scaling for real robot
                if use_real_robot:
                    scaled_linear = a_in[0] * linear_velocity_scale
                    scaled_angular = a_in[1] * angular_velocity_scale
                    
                    # Clamp to maximum velocities
                    scaled_linear = np.clip(scaled_linear, 0.0, max_linear_velocity)
                    scaled_angular = np.clip(scaled_angular, -max_angular_velocity, max_angular_velocity)
                    
                    cmd_linear = scaled_linear
                    cmd_angular = scaled_angular
                else:
                    cmd_linear = a_in[0]
                    cmd_angular = a_in[1]
                
                # Publish velocity commands to ROS
                twist_msg.linear.x = float(cmd_linear)
                twist_msg.angular.z = float(cmd_angular)
                cmd_vel_publisher.publish(twist_msg)

                print(f"cmd_x: {twist_msg.linear.x:.3f} (raw: {a_in[0]:.3f}), cmd_w: {twist_msg.angular.z:.3f} (raw: {a_in[1]:.3f})")
                
                # If in real robot mode and using real odometry, only use simulation for aspects not covered by real data
                if use_real_robot and ros_odom_received:
                    # We already have distance, goal status and collision from odometry above
                    # Just run step to keep simulation in sync, but don't override real-world values
                    _, _, _, _, _, _, a, reward = sim.step(
                        lin_velocity=a_in[0], ang_velocity=a_in[1]
                    )
                else:
                    # Pure simulation mode - use all simulation data
                    latest_scan, distance, cos, sin, collision, goal, a, reward = sim.step(
                        lin_velocity=a_in[0], ang_velocity=a_in[1]
                    )
                
                total_reward += reward
                total_steps += 1
                count += 1
                
                if goal:
                    print(f"Goal {idx+1} reached! Distance: {distance_to_goal:.3f}m")
                    goals += 1
                    break  # Move to next goal
                
                if collision:
                    print(f"Collision detected at distance: {min_scan_distance:.3f}m")
                    col += 1
                    if not ignore_collisions:
                        break  # Stop on collision only if not ignoring collisions
                
                # Small delay for real robot to avoid excessive CPU usage
                if use_real_robot:
                    rclpy.spin_once(ros_node, timeout_sec=0.05)  # 50ms delay
                
                done = goal or (collision and not ignore_collisions)
            
            if count >= current_max_steps and not done:
                print(f"Timeout reached for goal {idx+1}. Moving to next goal.")
            
            if collision and not ignore_collisions:
                print("Collision detected. Stopping path following.")
                break

        avg_step_reward = total_reward / total_steps if total_steps > 0 else 0
        avg_reward = total_reward / len(robot_goals)
        avg_col = col / len(robot_goals)
        avg_goal = goals / len(robot_goals)
        print(f"Total Reward: {total_reward}")
        print(f"Average Reward: {avg_reward}")
        print(f"Average Step Reward: {avg_step_reward}")
        print(f"Average Collision rate: {avg_col}")
        print(f"Average Goal rate: {avg_goal}")
        print("..............................................")
        model.writer.add_scalar("test/total_reward", total_reward, epoch)
        model.writer.add_scalar("test/avg_reward", avg_reward, epoch)
        model.writer.add_scalar("test/avg_step_reward", avg_step_reward, epoch)
        model.writer.add_scalar("test/avg_col", avg_col, epoch)
        model.writer.add_scalar("test/avg_goal", avg_goal, epoch)
    
    finally:
        # Clean up ROS resources
        ros_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
