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
    # URDF AND SIMULATION BOOTSTRAP
    # The robot model is loaded from the package share directory and converted
    # into a robot_description string that can be consumed by the ROS nodes.
    # ------------------------------------------------------------------
    urdf_file_path = os.path.join(
        get_package_share_directory('my_robot'),
        'urdf',
        'my_robot.xacro.urdf'
    )
    robot_description = cast(Any, xacro.process_file(urdf_file_path)).toxml()

    # ------------------------------------------------------------------
    # ROBOT STATE PUBLISHER
    # Broadcasts the robot TF tree so the rest of the system can resolve frames
    # such as base_link, wheel joints, and the torso from the URDF.
    # ------------------------------------------------------------------
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description}]
    )

    # ------------------------------------------------------------------
    # GAZEBO STARTUP
    # Gazebo initializes the world and the simulator plugins required for ROS-
    # aware control and state publishing.
    # ------------------------------------------------------------------
    start_gazebo_cmd = ExecuteProcess(
        cmd=['gazebo', '--verbose', '-s', 'libgazebo_ros_init.so', '-s', 'libgazebo_ros_factory.so'],
        output='screen'
    )

    # ------------------------------------------------------------------
    # MODEL SPAWNING
    # The robot entity is inserted into the running world after the simulator has
    # started, so Gazebo can create the correct sensors and links.
    # ------------------------------------------------------------------
    spawn_entity_node = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-entity', 'my_robot', '-topic', 'robot_description'],
        output='screen'
    )

    # ------------------------------------------------------------------
    # CONTROLLER SPAWNERS
    # These processes load the required ROS2 controllers only after the robot has
    # been spawned into the simulation environment.
    # ------------------------------------------------------------------
    spawn_joint_state_broadcaster = ExecuteProcess(
        cmd=['ros2', 'run', 'controller_manager', 'spawner', 'joint_state_broadcaster', '--controller-manager', '/controller_manager'],
        output='screen'
    )
    
    spawn_joint_trajectory_controller = ExecuteProcess(
        cmd=['ros2', 'run', 'controller_manager', 'spawner', 'joint_trajectory_controller', '--controller-manager', '/controller_manager'], # <-- NEW NAME
        output='screen'
    )

    return LaunchDescription([
        robot_state_publisher_node,
        start_gazebo_cmd,
        spawn_entity_node,
        
        # ------------------------------------------------------------------
        # EVENT HANDLERS
        # The controller spawners are started only after Gazebo has finished
        # placing the robot in the world so the hardware interfaces can register
        # with the controller manager without errors.
        # ------------------------------------------------------------------
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=spawn_entity_node,
                on_exit=[spawn_joint_state_broadcaster],
            )
        ),
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=spawn_joint_state_broadcaster,
                on_exit=[spawn_joint_trajectory_controller], # <-- NEW VARIABLE
            )
        ),
    ])