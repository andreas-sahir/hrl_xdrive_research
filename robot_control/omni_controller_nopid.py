#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from nav_msgs.msg import Odometry
import math
from typing import cast

class OmniController(Node):
    def __init__(self):
        super().__init__('omni_controller')

        # Robot parameters
        self.declare_parameter('wheel_radius', 0.05)
        self.declare_parameter('wheel_separation_x', 0.54)
        self.declare_parameter('wheel_separation_y', 0.54)

        wheel_radius_param = self.get_parameter('wheel_radius')
        wheel_sep_x_param = self.get_parameter('wheel_separation_x')
        wheel_sep_y_param = self.get_parameter('wheel_separation_y')

        self.r = float(wheel_radius_param.value if wheel_radius_param.value is not None else 0.05)
        self.l = float(wheel_sep_x_param.value if wheel_sep_x_param.value is not None else 0.54) / 2.0
        self.L = float(wheel_sep_y_param.value if wheel_sep_y_param.value is not None else 0.54) / 2.0

        self.SQRT_2 = math.sqrt(2.0)
        self.r_scaled = self.r * self.SQRT_2

        # PID Parameters
        self.declare_parameter('pid.vx.kp', 1.5)
        self.declare_parameter('pid.vy.kp', 1.5)
        self.declare_parameter('pid.wz.kp', 1.5)
        
        self.kp_vx = self.get_parameter('pid.vx.kp').value
        self.kp_vy = self.get_parameter('pid.vy.kp').value
        self.kp_wz = self.get_parameter('pid.wz.kp').value

        self.joint_names = [
            "front_left_wheel_joint", "front_right_wheel_joint",
            "back_left_wheel_joint", "back_right_wheel_joint"
        ]

        self.target_vx = 0.0; self.target_vy = 0.0; self.target_wz = 0.0
        
        self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_callback, 10)
        self.wheel_pub = self.create_publisher(JointTrajectory, '/joint_trajectory_controller/joint_trajectory', 10)
        self.timer = self.create_timer(0.05, self.update_loop) # 20Hz

        self.get_logger().info('Omni Controller (Fixed Mismatch) started.')

    def cmd_vel_callback(self, msg):
        self.target_vx = msg.linear.x
        self.target_vy = msg.linear.y
        self.target_wz = msg.angular.z

    def update_loop(self):
        # Simple Open Loop for stability (PID removed for now to isolate movement)
        vx, vy, wz = self.target_vx, self.target_vy, self.target_wz
        
        # Inverse Kinematics
        w1 = (1/self.r_scaled) * (vx - vy - (self.L + self.l) * wz) 
        w2 = (1/self.r_scaled) * (vx + vy + (self.L + self.l) * wz) 
        w3 = (1/self.r_scaled) * (vx + vy - (self.L + self.l) * wz) 
        w4 = (1/self.r_scaled) * (vx - vy + (self.L + self.l) * wz) 

        traj_msg = JointTrajectory()
        traj_msg.joint_names = self.joint_names
        point = JointTrajectoryPoint()
        
        # CRITICAL FIX: Provide dummy positions to stop the error
        point.positions = [0.0, 0.0, 0.0, 0.0] 
        point.velocities = [w1, w2, w3, w4]
        point.time_from_start.sec = 0
        point.time_from_start.nanosec = 50000000 # 50ms
        
        traj_msg.points = [point]
        self.wheel_pub.publish(traj_msg)

def main(args=None):
    rclpy.init(args=args)
    node = OmniController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()