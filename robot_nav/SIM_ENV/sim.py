import irsim
import numpy as np
import random
from collections import deque

from robot_nav.SIM_ENV.sim_env import SIM_ENV


class SIM(SIM_ENV):
    """
    A simulation environment interface for robot navigation using IRSim.

    This class wraps around the IRSim environment and provides methods for stepping,
    resetting, and interacting with a mobile robot, including reward computation.

    Attributes:
        env (object): The simulation environment instance from IRSim.
        robot_goal (np.ndarray): The goal position of the robot.
        prev_ang_velocity (float): Previous angular velocity for zigzag penalty.
        position_history (deque): Recent position history to detect circular motion.
        prev_distance_to_goal (float): Previous distance to goal for progress tracking.
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
        self.prev_ang_velocity = 0.0
        self.position_history = deque(maxlen=20)  # Keep last 20 positions
        self.prev_distance_to_goal = float('inf')

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
        current_position = [robot_state[0].item(), robot_state[1].item()]
        
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
        
        # Add current position to history
        self.position_history.append(current_position)
        
        reward = self.get_reward(
            goal, collision, action, latest_scan, 
            self.prev_ang_velocity, self.position_history,
            distance, self.prev_distance_to_goal
        )
        
        # Update previous values for next step
        self.prev_ang_velocity = ang_velocity
        self.prev_distance_to_goal = distance

        return latest_scan, distance, cos, sin, collision, goal, action, reward

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
            robot_state = [[random.uniform(1, 5)], [random.uniform(1, 5)], [0]]

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
                range_limits=[[1, 1, -3.141592653589793], [5, 5, 3.141592653589793]],
            )
        else:
            self.env.robot.set_goal(np.array(robot_goal), init=True)
        self.env.reset()
        self.robot_goal = self.env.robot.goal
        
        # Reset previous angular velocity
        self.prev_ang_velocity = 0.0
        self.position_history.clear()
        self.prev_distance_to_goal = float('inf')

        action = [0.0, 0.0]
        latest_scan, distance, cos, sin, _, _, action, reward = self.step(
            lin_velocity=action[0], ang_velocity=action[1]
        )
        return latest_scan, distance, cos, sin, False, False, action, reward

    @staticmethod
    def get_reward(goal, collision, action, laser_scan, prev_ang_velocity, 
                   position_history, current_distance, prev_distance):
        """
        Calculate the reward for the current step.

        Args:
            goal (bool): Whether the goal has been reached.
            collision (bool): Whether a collision occurred.
            action (list): The action taken [linear velocity, angular velocity].
            laser_scan (list): The LIDAR scan readings.
            prev_ang_velocity (float): Previous angular velocity for zigzag penalty.
            position_history (deque): Recent position history to detect circular motion.
            current_distance (float): Current distance to goal.
            prev_distance (float): Previous distance to goal.

        Returns:
            (float): Computed reward for the current state.
        """
        if goal:
            return 100.0
        elif collision:
            return -100.0
        else:
            r3 = lambda x: 1.35 - x if x < 1.35 else 0.0

            # Encourage linear velocity around 1.0 (peak at 1.0, penalty for deviations)
            linear_reward = action[0] - 0.5 * (action[0] - 1.0) ** 2

            # Encourage angular velocity towards -2 or 2 (penalty for values near 0)
            ang_vel = action[1]
            if abs(ang_vel) < 1.5:
                # Penalty for being in the deadband region
                angular_penalty = abs(ang_vel) * 0.5
            else:
                # Reward for being near -2 or 2
                angular_penalty = -0.3 * (1 - abs(abs(ang_vel) - 2.0))
            
            # Zigzag penalty: penalize rapid changes in angular velocity
            ang_vel_change = abs(ang_vel - prev_ang_velocity)
            zigzag_penalty = ang_vel_change * 0.3
            
            # Circular motion penalty: check if robot is revisiting previous positions
            circular_penalty = 0.0
            if len(position_history) >= 10:
                current_pos = position_history[-1]
                # Check distance to older positions in history
                for old_pos in list(position_history)[:-5]:  # Exclude very recent positions
                    dist_to_old = np.linalg.norm(np.array(current_pos) - np.array(old_pos))
                    if dist_to_old < 0.5:  # If within 0.5m of a previous position
                        circular_penalty += 0.5
            
            # Progress penalty: penalize if not making progress toward goal
            progress_reward = (prev_distance - current_distance) * 2.0  # Reward for getting closer
            
            return (linear_reward - angular_penalty - zigzag_penalty - 
                   circular_penalty + progress_reward - r3(min(laser_scan)) / 2)

    def set_robot_goal(self, robot_goal):
        """Set a new goal for the robot in the environment.

        Args:
            robot_goal (list or np.ndarray): The new goal position for the robot.
        """
        self.env.robot.set_goal(np.array(robot_goal), init=False)
        self.robot_goal = self.env.robot.goal
