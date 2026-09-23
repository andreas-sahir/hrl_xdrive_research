import os
from typing import Any, cast
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch_ros.actions import Node
import xacro

def generate_launch_description():
    # ------------------------------------------------------------------
    # PATHS AND URDF PREPARATION
    # The package share directory is resolved first so the launch file can find
    # the URDF and EKF configuration files in a ROS2 package layout.
    # ------------------------------------------------------------------
    pkg_path = get_package_share_directory('my_robot')
    urdf_file = os.path.join(pkg_path, 'urdf', 'my_robot.xacro.urdf')
    ekf_config_path = os.path.join(pkg_path, 'config', 'ekf.yaml')

    # ------------------------------------------------------------------
    # ROBOT DESCRIPTION PARSING
    # The Xacro description is converted into a single XML robot model that the
    # simulator and state publisher can consume.
    # ------------------------------------------------------------------
    doc = cast(Any, xacro.process_file(urdf_file))
    robot_desc = doc.toxml()

    # ------------------------------------------------------------------
    # HEADLESS GAZEBO SERVER
    # The simulation is launched without a GUI, which is useful for automated
    # testing, training, and CI-like CPU-only environments.
    # ------------------------------------------------------------------
    start_gazebo_server = ExecuteProcess(
        cmd=['gzserver', '--verbose', '-s', 'libgazebo_ros_init.so', '-s', 'libgazebo_ros_factory.so'],
        output='screen'
    )

    # ------------------------------------------------------------------
    # STATE PUBLISHING AND SPAWNING
    # The robot state publisher exposes TF frames and the model is inserted into
    # the world so sensor and actuator plugins can connect to the same URDF.
    # ------------------------------------------------------------------
    node_robot_state_pub = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_desc}]
    )

    node_spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-topic', 'robot_description', '-entity', 'my_robot'],
        output='screen'
    )

    # ------------------------------------------------------------------
    # HARDWARE INTERFACES AND FILTERING
    # These spawners match the controller names defined in the robot YAML config
    # and start the EKF/odom pipeline that smooths noisy motion estimation.
    # ------------------------------------------------------------------
    spawn_jsb = ExecuteProcess(
        cmd=['ros2', 'run', 'controller_manager', 'spawner', 'joint_state_broadcaster'],
        output='screen'
    )
    spawn_velocity_controller = ExecuteProcess(
        cmd=['ros2', 'run', 'controller_manager', 'spawner', 'velocity_controller'],
        output='screen'
    )

    node_ekf = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node',
        output='screen',
        parameters=[ekf_config_path]
    )

    node_odom_pub = Node(
        package='my_robot',
        executable='odometry_publisher.py',
        name='odometry_publisher',
        parameters=[{'publish_tf': True}]
    )


    return LaunchDescription([
        start_gazebo_server,
        node_robot_state_pub,
        node_spawn_entity,
        node_ekf,
        node_odom_pub,
        
        RegisterEventHandler(
            event_handler=OnProcessExit(target_action=node_spawn_entity, on_exit=[spawn_jsb])
        ),
        RegisterEventHandler(
            event_handler=OnProcessExit(target_action=spawn_jsb, on_exit=[spawn_velocity_controller]) # Updated here
        )
    ])