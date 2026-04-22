#!/usr/bin/env python3

from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import math

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ModelStates


class DefenderController(Node):
    def __init__(self):
        super().__init__('defender_controller')

        # Publisher to move the robot
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Subscriber to Gazebo model states
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE
        )

        self.model_sub = self.create_subscription(
            ModelStates,
            '/gazebo/model_states',
            self.model_states_callback,
            qos
        )

        # Store current positions
        self.robot_x = None
        self.robot_y = None
        self.robot_yaw = None

        self.scorer_x = None
        self.scorer_y = None

        # Fixed goal position
        self.goal_x = 5.0
        self.goal_y = 0.0

        # Control loop timer
        self.timer = self.create_timer(0.1, self.control_loop)

        self.get_logger().info('Defender controller started.')

    def quaternion_to_yaw(self, x, y, z, w):
        """Convert quaternion to yaw angle."""
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    def model_states_callback(self, msg):
        """Read robot and ball positions from Gazebo."""
        self.get_logger().info(f'Model names: {msg.name}')

        try:
            robot_index = msg.name.index('defender')
            robot_pose = msg.pose[robot_index]

            self.robot_x = robot_pose.position.x
            self.robot_y = robot_pose.position.y
            self.robot_yaw = self.quaternion_to_yaw(
                robot_pose.orientation.x,
                robot_pose.orientation.y,
                robot_pose.orientation.z,
                robot_pose.orientation.w
            )
        except ValueError:
            self.get_logger().info('Could not find model name: defender')

        try:
            scorer_index = msg.name.index('scorer')
            scorer_pose = msg.pose[scorer_index]

            self.scorer_x = scorer_pose.position.x
            self.scorer_y = scorer_pose.position.y
        except ValueError:
            self.get_logger().info('Could not find model name: scorer')

    def control_loop(self):
        """Move robot toward a blocking point between ball and goal."""
        if self.robot_x is None or self.scorer_x is None:
            self.get_logger().info(
                f'Waiting for data: robot_x={self.robot_x}, ball_x={self.scorer_x}'
            )
            return

        # Vector from ball to goal
        dx = self.goal_x - self.scorer_x
        dy = self.goal_y - self.scorer_y
        dist_bg = math.sqrt(dx * dx + dy * dy)

        if dist_bg < 1e-6:
            return

        # Unit vector from ball to goal
        ux = dx / dist_bg
        uy = dy / dist_bg

        # Blocking point: some distance from the ball toward the goal
        block_distance = 0.8
        target_x = self.scorer_x + block_distance * ux
        target_y = self.scorer_y + block_distance * uy

        # Vector from robot to target
        error_x = target_x - self.robot_x
        error_y = target_y - self.robot_y
        distance_to_target = math.sqrt(error_x * error_x + error_y * error_y)

        # Desired heading
        target_heading = math.atan2(error_y, error_x)
        heading_error = target_heading - self.robot_yaw

        # Wrap angle to [-pi, pi]
        while heading_error > math.pi:
            heading_error -= 2.0 * math.pi
        while heading_error < -math.pi:
            heading_error += 2.0 * math.pi

        cmd = Twist()

        # Simple proportional control
        if distance_to_target > 0.05:
            cmd.angular.z = 2.5 * heading_error

            # Move forward only if mostly facing target
            if abs(heading_error) < 0.6:
                cmd.linear.x = min(0.6, 1.0 * distance_to_target)
            else:
                cmd.linear.x = 0.0
        else:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0

        self.get_logger().info(
            f'robot=({self.robot_x:.2f},{self.robot_y:.2f}) '
            f'ball=({self.scorer_x:.2f},{self.scorer_y:.2f}) '
            f'target=({target_x:.2f},{target_y:.2f}) '
            f'dist={distance_to_target:.2f} '
            f'heading_error={heading_error:.2f}'
        )

        self.cmd_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = DefenderController()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    stop_msg = Twist()
    node.cmd_pub.publish(stop_msg)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
