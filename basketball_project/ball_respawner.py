#!/usr/bin/env python3

import os
import random

import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SpawnEntity, DeleteEntity
from ament_index_python.packages import get_package_share_directory


class BallRespawner(Node):
    def __init__(self):
        super().__init__('ball_respawner')

        self.spawn_client = self.create_client(SpawnEntity, '/spawn_entity')
        self.delete_client = self.create_client(DeleteEntity, '/delete_entity')

        while not self.spawn_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /spawn_entity service...')

        while not self.delete_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /delete_entity service...')

        package_share = get_package_share_directory('basketball_project')
        self.model_path = os.path.join(package_share, 'models', 'basketball', 'model.sdf')

        with open(self.model_path, 'r') as f:
            self.ball_xml = f.read()

        self.has_spawned_initial = False
        self.spawn_delay_timer = None

        self.initial_timer = self.create_timer(0.5, self.initial_spawn)
        self.respawn_timer = self.create_timer(10.0, self.respawn_ball)

        self.get_logger().info('Ball respawner started.')

    def random_position(self):
        x = random.uniform(-2.0, 2.0)
        y = random.uniform(0.5, 2.5)
        return x, y

    def spawn_ball(self):
        x, y = self.random_position()

        req = SpawnEntity.Request()
        req.name = 'basketball'
        req.xml = self.ball_xml
        req.initial_pose.position.x = x
        req.initial_pose.position.y = y
        req.initial_pose.position.z = 0.2

        future = self.spawn_client.call_async(req)

        def on_done(_future):
            self.get_logger().info(f'Spawned basketball at ({x:.2f}, {y:.2f})')

        future.add_done_callback(on_done)

    def initial_spawn(self):
        if self.has_spawned_initial:
            return
        self.has_spawned_initial = True
        self.initial_timer.cancel()
        self.spawn_ball()

    def respawn_ball(self):
        delete_req = DeleteEntity.Request()
        delete_req.name = 'basketball'

        delete_future = self.delete_client.call_async(delete_req)
        delete_future.add_done_callback(self._after_delete)

    def _after_delete(self, _future):
        self.get_logger().info('Deleted old basketball.')

        # short delay before respawn so Gazebo fully clears the old entity
        self.spawn_delay_timer = self.create_timer(0.5, self._delayed_spawn)

    def _delayed_spawn(self):
        self.spawn_delay_timer.cancel()
        self.spawn_delay_timer = None
        self.spawn_ball()


def main(args=None):
    rclpy.init(args=args)
    node = BallRespawner()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
