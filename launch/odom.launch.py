import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    # Path to your working simulation launch file
    simulation_launch_path = os.path.join(
        get_package_share_directory('my_robot'),
        'launch',
        'simulation.launch.py' # This must be the name of your working launch file
    )

    # Path to your EKF configuration file (we will create this in the next step)
    ekf_config_path = os.path.join(
        get_package_share_directory('my_robot'),
        'config',
        'ekf.yaml'
    )

    # Include your existing simulation launch
    start_simulation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(simulation_launch_path)
    )

    # Start the robot_localization (EKF) node
    start_robot_localization_node = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node',
        output='screen',
        parameters=[ekf_config_path]
    )
    start_odometry_publisher_node = Node(
        package='my_robot',
        executable='odometry_publisher.py', # Make sure this script is executable (chmod +x)
        name='odometry_publisher'
    )

    return LaunchDescription([
        start_simulation,
        start_robot_localization_node,
        start_odometry_publisher_node,
    ])