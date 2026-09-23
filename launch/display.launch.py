import os
from typing import Any, cast
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess, RegisterEventHandler, DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.event_handlers import OnProcessExit
from launch_ros.actions import Node
import xacro

def generate_launch_description():
    pkg_path = get_package_share_directory('my_robot')

    # --- 1. SETUP PATHS ---
    default_world_path = os.path.join(pkg_path, 'worlds', 'turtlebot3_dqn_stage3.world') #chane to 1, 2, 3, or 4
    
    urdf_file = os.path.join(pkg_path, 'urdf', 'my_robot.xacro.urdf')
    doc = cast(Any, xacro.process_file(urdf_file))
    robot_desc = doc.toxml()

    # --- 2. ARGUMENTS ---
    world_arg = DeclareLaunchArgument(
        'world',
        default_value=default_world_path,
        description='Path to the world file to load'
    )

    # --- 3. GAZEBO NODES ---
    start_gazebo_cmd = ExecuteProcess(
        cmd=['gazebo', '--verbose', LaunchConfiguration('world'), 
             '-s', 'libgazebo_ros_init.so', 
             '-s', 'libgazebo_ros_factory.so'],
        output='screen'
    )

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_desc}]
    )

    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-entity', 'my_robot', '-topic', 'robot_description', '-z', '0.1'],
        output='screen'
    )

    return LaunchDescription([
        world_arg,
        start_gazebo_cmd,
        robot_state_publisher,
        spawn_entity
    ])