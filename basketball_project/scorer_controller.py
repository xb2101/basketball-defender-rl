#!/usr/bin/env python3

import math
import random

import rclpy
from rclpy.node import Node

from gazebo_msgs.msg import ModelStates
from gazebo_msgs.srv import SetEntityState
from gazebo_msgs.msg import EntityState


class ScorerController(Node):
    def __init__(self):
        super().__init__('scorer_controller')

        self.set_state_client = self.create_client(SetEntityState, '/gazebo/set_entity_state')
        while not self.set_state_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /gazebo/set_entity_state service...')

        self.model_sub = self.create_subscription(
            ModelStates,
            '/gazebo/model_states',
            self.model_states_callback,
            10
        )

        self.scorer_name = 'scorer'
        self.scorer_x = None
        self.scorer_y = None

        # Stages:
        # 0 = moving to random court positions
        # 1 = driving to paint
        self.stage = 0
        self.random_stops_remaining = 0

        self.current_target_x = None
        self.current_target_y = None

        self.speed = 0.3
        self.waypoint_tolerance = 0.3

        self.initialized_position = False

        self.timer = self.create_timer(0.02, self.control_loop)
        self.get_logger().info('Scorer controller started.')

    def _pick_random_court_position(self):
        """Pick a random position around the court away from the paint."""
        x = random.uniform(0.5, 3.5)
        y = random.uniform(-3.5, 3.5)
        return x, y

    def _new_episode(self):
        """Start a new attacking sequence."""
        # Visit 2-3 random spots before attacking
        self.random_stops_remaining = random.randint(2, 3)
        self.stage = 0
        tx, ty = self._pick_random_court_position()
        self.current_target_x = tx
        self.current_target_y = ty

    def model_states_callback(self, msg):
        try:
            idx = msg.name.index(self.scorer_name)
            pose = msg.pose[idx]
            self.scorer_x = pose.position.x
            self.scorer_y = pose.position.y
        except ValueError:
            pass

    def move_scorer(self, x, y):
        req = SetEntityState.Request()
        state = EntityState()
        state.name = self.scorer_name
        state.pose.position.x = x
        state.pose.position.y = y
        state.pose.position.z = 0.2
        state.pose.orientation.w = 1.0
        state.twist.linear.x = 0.0
        state.twist.linear.y = 0.0
        state.twist.linear.z = 0.0
        state.twist.angular.x = 0.0
        state.twist.angular.y = 0.0
        state.twist.angular.z = 0.0
        req.state = state
        self.set_state_client.call_async(req)

    def control_loop(self):
        if self.scorer_x is None or self.scorer_y is None:
            return

        if not self.initialized_position:
            self._new_episode()
            self.initialized_position = True
            return

        if self.stage == 0:
            target_x = self.current_target_x
            target_y = self.current_target_y
        else:
            target_x = 5.5
            target_y = 0.0

        dx = target_x - self.scorer_x
        dy = target_y - self.scorer_y
        dist = math.sqrt(dx * dx + dy * dy)

        if dist < self.waypoint_tolerance or (self.stage == 1 and self.scorer_x >= 4.5):
            if self.stage == 0:
                self.random_stops_remaining -= 1
                if self.random_stops_remaining <= 0:
                    self.stage = 1
                else:
                    tx, ty = self._pick_random_court_position()
                    self.current_target_x = tx
                    self.current_target_y = ty
            else:
                self._new_episode()
            return

        ux = dx / dist
        uy = dy / dist

        dt = 0.02
        new_x = self.scorer_x + self.speed * dt * ux
        new_y = self.scorer_y + self.speed * dt * uy
        self.move_scorer(new_x, new_y)


def main(args=None):
    rclpy.init(args=args)
    node = ScorerController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()