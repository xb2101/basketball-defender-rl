from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction, SetEnvironmentVariable
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    package_name = 'basketball_project'
    package_share = get_package_share_directory(package_name)

    world_path = os.path.join(package_share, 'worlds', 'basketball_court.world')
    models_path = os.path.join(package_share, 'models')
    scorer_model_path = os.path.join(package_share, 'models', 'basketball', 'model.sdf')
    turtlebot_model_path = '/opt/ros/foxy/share/turtlebot3_gazebo/models/turtlebot3_burger/model.sdf'

    set_tb3_model = SetEnvironmentVariable(
        name='TURTLEBOT3_MODEL',
        value='burger'
    )

    set_gazebo_model_path = SetEnvironmentVariable(
        name='GAZEBO_MODEL_PATH',
        value=f'/opt/ros/foxy/share/turtlebot3_gazebo/models:{models_path}'
    )

    set_gazebo_plugin_path = SetEnvironmentVariable(
        name='GAZEBO_PLUGIN_PATH',
        value='/opt/ros/foxy/lib'
    )

    gazebo = ExecuteProcess(
        cmd=[
            'gazebo',
            world_path,
            '-s', '/opt/ros/foxy/lib/libgazebo_ros_init.so',
            '-s', '/opt/ros/foxy/lib/libgazebo_ros_factory.so'
        ],
        output='screen'
    )

    spawn_scorer = TimerAction(
        period=2.0,
        actions=[
            ExecuteProcess(
                cmd=[
                    'ros2', 'run', 'gazebo_ros', 'spawn_entity.py',
                    '-file', scorer_model_path,
                    '-entity', 'scorer',
                    '-x', '1.0',
                    '-y', '0.0',
                    '-z', '0.2'
                ],
                output='screen'
            )
        ]
    )

    spawn_defender = TimerAction(
        period=4.0,
        actions=[
            ExecuteProcess(
                cmd=[
                    'ros2', 'run', 'gazebo_ros', 'spawn_entity.py',
                    '-file', turtlebot_model_path,
                    '-entity', 'defender',
                    '-x', '2.0',
                    '-y', '0.0',
                    '-z', '0.1'
                ],
                output='screen'
            )
        ]
    )

    start_scorer_controller = TimerAction(
        period=6.0,
        actions=[
            Node(
                package='basketball_project',
                executable='scorer_controller',
                output='screen'
            )
        ]
    )

    return LaunchDescription([
        set_tb3_model,
        set_gazebo_model_path,
        set_gazebo_plugin_path,
        gazebo,
        spawn_scorer,
        spawn_defender,
        start_scorer_controller,
    ])
