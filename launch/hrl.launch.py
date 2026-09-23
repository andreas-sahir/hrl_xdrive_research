import os
from typing import Any, cast
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess, RegisterEventHandler, DeclareLaunchArgument
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition, UnlessCondition
from launch_ros.actions import Node
import xacro

def generate_launch_description():
    pkg_path = get_package_share_directory('my_robot')
    urdf_file = os.path.join(pkg_path, 'urdf', 'my_robot.xacro.urdf')
    ekf_config_path = os.path.join(pkg_path, 'config', 'ekf.yaml')
    
    doc = cast(Any, xacro.process_file(urdf_file))
    robot_desc = doc.toxml()

    # --- LAUNCH ARGUMENTS ---
    declare_gui = DeclareLaunchArgument('use_gui', default_value='false', description='Set to true to launch Gazebo GUI')
    declare_world = DeclareLaunchArgument('world', default_value=os.path.join(pkg_path, 'worlds', 'turtlebot3_dqn_stage1.world'), description='Target world')
    
    use_gui = LaunchConfiguration('use_gui')
    world_file = LaunchConfiguration('world')

    # --- GAZEBO CONDITIONAL LAUNCH ---
    start_gazebo_headless = ExecuteProcess(
        condition=UnlessCondition(use_gui),
        cmd=['gzserver', '--verbose', '-s', 'libgazebo_ros_init.so', '-s', 'libgazebo_ros_factory.so', '-s', 'libgazebo_ros_reset.so', world_file],
        output='screen'
    )
    
    start_gazebo_gui = ExecuteProcess(
        condition=IfCondition(use_gui),
        cmd=['gazebo', '--verbose', '-s', 'libgazebo_ros_init.so', '-s', 'libgazebo_ros_factory.so', '-s', 'libgazebo_ros_reset.so', world_file],
        output='screen'
    )

    # --- CORE NODES ---
    node_robot_state_pub = Node(package='robot_state_publisher', executable='robot_state_publisher', output='screen', parameters=[{'robot_description': robot_desc, 'use_sim_time': True}])
    node_spawn_entity = Node(package='gazebo_ros', executable='spawn_entity.py', arguments=['-topic', 'robot_description', '-entity', 'my_robot'], output='screen')
    spawn_jsb = ExecuteProcess(cmd=['ros2', 'run', 'controller_manager', 'spawner', 'joint_state_broadcaster'], output='screen')
    spawn_jtc = ExecuteProcess(cmd=['ros2', 'run', 'controller_manager', 'spawner', 'joint_trajectory_controller'], output='screen')
    node_ekf = Node(package='robot_localization', executable='ekf_node', name='ekf_filter_node', parameters=[ekf_config_path])
    node_odom_pub = Node(package='my_robot', executable='odometry_publisher.py', parameters=[{'publish_tf': True, 'use_sim_time': True}])
    node_omni_nopid = Node(package='my_robot', executable='omni_controller_nopid.py', output='screen')

    return LaunchDescription([
        declare_gui, declare_world,
        start_gazebo_headless, start_gazebo_gui,
        node_robot_state_pub, node_spawn_entity, node_ekf, node_odom_pub, node_omni_nopid,
        RegisterEventHandler(event_handler=OnProcessExit(target_action=node_spawn_entity, on_exit=[spawn_jsb])),
        RegisterEventHandler(event_handler=OnProcessExit(target_action=spawn_jsb, on_exit=[spawn_jtc])),
    ])