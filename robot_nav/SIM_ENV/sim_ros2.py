import irsim
import numpy as np
import random
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion, Point, Pose, Twist, Vector3
from tf_transformations import quaternion_from_euler
from std_msgs.msg import Header
import time

from robot_nav.SIM_ENV.sim_env import SIM_ENV


class SIM(SIM_ENV):
    """
    A simulation environment interface for robot navigation using IRSim.

    This class wraps around the IRSim environment and provides methods for stepping,
    resetting, and interacting with a mobile robot, including reward computation.

    Attributes:
        env (object): The simulation environment instance from IRSim.
        robot_goal (np.ndarray): The goal position of the robot.
    """

    def __init__(self, world_file="robot_world.yaml", disable_plotting=False):
        """
        Initialize the simulation environment.

        Args:
            world_file (str): Path to the world configuration YAML file.
            disable_plotting (bool): If True, disables rendering and plotting.
        """
        display = False if disable_plotting else True
        self.env = irsim.make(
            world_file, disable_all_plot=disable_plotting, display=display
        )
        robot_info = self.env.get_robot_info(0)
        self.robot_goal = robot_info.goal

        # Initialize ROS2
        rclpy.init(args=None)
        self.node = Node("irsim_interface")

        # Create publishers
        self.laser_pub = self.node.create_publisher(LaserScan, "scan", 10)
        self.odom_pub = self.node.create_publisher(Odometry, "odom", 10)

        # Initialize message timestamps
        self.seq = 0

    def step(self, lin_velocity=0.0, ang_velocity=0.1):
        """
        Perform one step in the simulation using the given control commands.

        Args:
            lin_velocity (float): Linear velocity to apply to the robot.
            ang_velocity (float): Angular velocity to apply to the robot.

        Returns:
            (tuple): Contains the latest LIDAR scan, distance to goal, cosine and sine of angle to goal,
                   collision flag, goal reached flag, applied action, and computed reward.
        """
        self.env.step(action_id=0, action=np.array([[lin_velocity], [ang_velocity]]))
        self.env.render()

        scan = self.env.get_lidar_scan()
        latest_scan = scan["ranges"]

        robot_state = self.env.get_robot_state()
        goal_vector = [
            self.robot_goal[0].item() - robot_state[0].item(),
            self.robot_goal[1].item() - robot_state[1].item(),
        ]
        distance = np.linalg.norm(goal_vector)
        goal = self.env.robot.arrive
        pose_vector = [np.cos(robot_state[2]).item(), np.sin(robot_state[2]).item()]
        cos, sin = self.cossin(pose_vector, goal_vector)
        collision = self.env.robot.collision
        action = [lin_velocity, ang_velocity]
        reward = self.get_reward(goal, collision, action, latest_scan)

        # Create fixed angles for 7 rays spanning 180 degrees
        angles = np.linspace(-np.pi / 2, np.pi / 2, 7)

        # Publish laser scan to ROS2 with fixed parameters
        self._publish_laser_scan(
            latest_scan, angles, 0.0, 8.0
        )

        # Publish odometry to ROS2
        self._publish_odometry(robot_state, lin_velocity, ang_velocity)

        # Process any pending ROS callbacks
        rclpy.spin_once(self.node, timeout_sec=0)

        return latest_scan, distance, cos, sin, collision, goal, action, reward

    def _publish_laser_scan(self, ranges, angles, range_min, range_max):
        """
        Publish laser scan data to ROS2 topic
        """
        msg = LaserScan()
        now = self.node.get_clock().now().to_msg()
        msg.header = Header(stamp=now, frame_id="laser")
        msg.angle_min = float(angles[0])
        msg.angle_max = float(angles[-1])
        msg.angle_increment = (
            float(angles[1] - angles[0]) if len(angles) > 1 else 0.0
        )
        msg.time_increment = 0.0
        msg.scan_time = 0.1
        msg.range_min = float(range_min)
        msg.range_max = float(range_max)
        msg.ranges = [float(r) for r in ranges]

        self.laser_pub.publish(msg)

    def _publish_odometry(self, robot_state, lin_vel, ang_vel):
        """
        Publish odometry data to ROS2 topic
        """
        msg = Odometry()
        now = self.node.get_clock().now().to_msg()
        msg.header = Header(stamp=now, frame_id="odom")
        msg.child_frame_id = "base_link"

        # Set position
        msg.pose.pose.position = Point(
            x=float(robot_state[0]), y=float(robot_state[1]), z=0.0
        )

        # Set orientation (convert Euler angle to quaternion)
        q = quaternion_from_euler(0, 0, float(robot_state[2]))
        msg.pose.pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])

        # Set linear and angular velocities
        msg.twist.twist.linear = Vector3(x=float(lin_vel), y=0.0, z=0.0)
        msg.twist.twist.angular = Vector3(x=0.0, y=0.0, z=float(ang_vel))

        self.odom_pub.publish(msg)

    def close(self):
        """
        Cleanup resources including ROS2 node
        """
        self.node.destroy_node()
        rclpy.shutdown()

    def reset(
        self,
        robot_state=None,
        robot_goal=None,
        random_obstacles=True,
        random_obstacle_ids=None,
    ):
        """
        Reset the simulation environment, optionally setting robot and obstacle states.

        Args:
            robot_state (list or None): Initial state of the robot as a list of [x, y, theta, speed].
            robot_goal (list or None): Goal state for the robot.
            random_obstacles (bool): Whether to randomly reposition obstacles.
            random_obstacle_ids (list or None): Specific obstacle IDs to randomize.

        Returns:
            (tuple): Initial observation after reset, including LIDAR scan, distance, cos/sin,
                   and reward-related flags and values.
        """
        if robot_state is None:
            robot_state = [[random.uniform(1, 9)], [random.uniform(1, 9)], [0]]

        self.env.robot.set_state(
            state=np.array(robot_state),
            init=True,
        )

        if random_obstacles:
            if random_obstacle_ids is None:
                random_obstacle_ids = [i + 1 for i in range(7)]
            self.env.random_obstacle_position(
                range_low=[0, 0, -3.14],
                range_high=[10, 10, 3.14],
                ids=random_obstacle_ids,
                non_overlapping=True,
            )

        if robot_goal is None:
            self.env.robot.set_random_goal(
                obstacle_list=self.env.obstacle_list,
                init=True,
                range_limits=[[1, 1, -3.141592653589793], [9, 9, 3.141592653589793]],
            )
        else:
            self.env.robot.set_goal(np.array(robot_goal), init=True)
        self.env.reset()
        self.robot_goal = self.env.robot.goal

        action = [0.0, 0.0]
        latest_scan, distance, cos, sin, _, _, action, reward = self.step(
            lin_velocity=action[0], ang_velocity=action[1]
        )
        return latest_scan, distance, cos, sin, False, False, action, reward

    @staticmethod
    def get_reward(goal, collision, action, laser_scan):
        """
        Calculate the reward for the current step.

        Args:
            goal (bool): Whether the goal has been reached.
            collision (bool): Whether a collision occurred.
            action (list): The action taken [linear velocity, angular velocity].
            laser_scan (list): The LIDAR scan readings.

        Returns:
            (float): Computed reward for the current state.
        """
        if goal:
            return 100.0
        elif collision:
            return -100.0
        else:
            r3 = lambda x: 1.35 - x if x < 1.35 else 0.0
            return action[0] - abs(action[1]) / 2 - r3(min(laser_scan)) / 2
