#!/usr/bin/env python3

import rclpy
import random
from gazebo_msgs.srv import SpawnEntity


def main():

    rclpy.init()

    node = rclpy.create_node('random_ball_spawner')

    client = node.create_client(SpawnEntity, '/spawn_entity')

    while not client.wait_for_service(timeout_sec=1.0):
        node.get_logger().info('Waiting for /spawn_entity service...')

    request = SpawnEntity.Request()

    # Load basketball model
    with open('/home/xavierb22/turtlebot3_ws/src/basketball_project/models/basketball/model.sdf', 'r') as f:
        request.xml = f.read()

    request.name = 'basketball'

    # RANDOM POSITION
    x = random.uniform(-2.0, 2.0)
    y = random.uniform(0.5, 2.5)

    request.initial_pose.position.x = x
    request.initial_pose.position.y = y
    request.initial_pose.position.z = 0.2

    node.get_logger().info(f'Spawning basketball at ({x:.2f}, {y:.2f})')

    future = client.call_async(request)
    rclpy.spin_until_future_complete(node, future)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
