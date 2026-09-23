#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from nav_msgs.msg import Odometry
import math

class PIDController:
    def __init__(self, kp, ki, kd, max_output, min_output, integral_limit=10.0):
        self.integral_limit = integral_limit
        self.kp = kp; self.ki = ki; self.kd = kd
        self.max_output = max_output; self.min_output = min_output
        self.integral = 0.0; self.previous_error = 0.0
        
    def calculate(self, setpoint, process_variable, dt):
        if dt <= 0.0: return 0.0 
        error = setpoint - process_variable
        p_term = self.kp * error
        self.integral += error * dt
        self.integral = max(min(self.integral, self.integral_limit), -self.integral_limit)
        i_term = self.ki * self.integral
        derivative = (error - self.previous_error) / dt
        d_term = self.kd * derivative
        self.previous_error = error
        output = p_term + i_term + d_term
        return max(min(output, self.max_output), self.min_output)

class OmniController(Node):
    def __init__(self):
        super().__init__('omni_controller')

        self.declare_parameter('wheel_radius', 0.05)
        self.declare_parameter('wheel_separation_x', 0.54)
        self.declare_parameter('wheel_separation_y', 0.54)
        self.r = float(self.get_parameter('wheel_radius').value)
        self.l = float(self.get_parameter('wheel_separation_x').value) / 2.0
        self.L = float(self.get_parameter('wheel_separation_y').value) / 2.0

        self.SQRT_2 = math.sqrt(2.0)
        self.r_scaled = self.r * self.SQRT_2

        # ------------------------------------------------------------------
        # PID GAINS FOR OMNI MOTION CONTROL
        # The gains are tuned separately for longitudinal velocity, lateral
        # velocity, and yaw rate so the controller can track a commanded planar
        # twist while keeping the robot stable under fast direction changes.
        # ------------------------------------------------------------------
        self.declare_parameter('pid.vx.kp', 1.5)
        self.declare_parameter('pid.vx.ki', 0.01)
        self.declare_parameter('pid.vx.kd', 0.1)
        self.declare_parameter('pid.vy.kp', 1.5)
        self.declare_parameter('pid.vy.ki', 0.01)
        self.declare_parameter('pid.vy.kd', 0.1)
        self.declare_parameter('pid.wz.kp', 1.5)
        self.declare_parameter('pid.wz.ki', 0.01)
        self.declare_parameter('pid.wz.kd', 0.1)

        self.joint_names = ["front_left_wheel_joint", "front_right_wheel_joint", "back_left_wheel_joint", "back_right_wheel_joint"]
        self.current_vx = 0.0; self.current_vy = 0.0; self.current_wz = 0.0
        self.target_vx = 0.0; self.target_vy = 0.0; self.target_wz = 0.0
        self.last_time = self.get_clock().now()

        # ------------------------------------------------------------------
        # COMMAND LIMITS
        # The controller keeps all motion commands within a ±1.0 m/s envelope to
        # keep the wheel trajectory smooth and within the robot's operational
        # limits.
        # ------------------------------------------------------------------
        VEL_MAX = 1.0
        VEL_MIN = -1.0

        self.pid_vx = PIDController(self.get_parameter('pid.vx.kp').value, self.get_parameter('pid.vx.ki').value, self.get_parameter('pid.vx.kd').value, VEL_MAX, VEL_MIN)
        self.pid_vy = PIDController(self.get_parameter('pid.vy.kp').value, self.get_parameter('pid.vy.ki').value, self.get_parameter('pid.vy.kd').value, VEL_MAX, VEL_MIN)
        self.pid_wz = PIDController(self.get_parameter('pid.wz.kp').value, self.get_parameter('pid.wz.ki').value, self.get_parameter('pid.wz.kd').value, 1.0, -1.0)
        
        self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_callback, 10)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.wheel_pub = self.create_publisher(JointTrajectory, '/joint_trajectory_controller/joint_trajectory', 10)
        self.timer = self.create_timer(0.02, self.update_loop)
        self.get_logger().info('Omni Controller PID (Delta Plan: 1.0 m/s) Started.')

    def cmd_vel_callback(self, msg):
        self.target_vx = msg.linear.x
        self.target_vy = msg.linear.y
        self.target_wz = msg.angular.z

    def odom_callback(self, msg):
        self.current_vx =msg.twist.twist.linear.x
        self.current_vy =msg.twist.twist.linear.y
        self.current_wz =msg.twist.twist.angular.z
    
    def update_loop(self):
        current_time = self.get_clock().now()
        dt = (current_time - self.last_time).nanoseconds / 1.0e9
        self.last_time = current_time
        if dt <= 0 : return

        vx_corrected = self.pid_vx.calculate(self.target_vx, self.current_vx, dt)
        vy_corrected = self.pid_vy.calculate(self.target_vy, self.current_vy, dt)
        wz_corrected = self.pid_wz.calculate(self.target_wz, self.current_wz, dt)

        # ------------------------------------------------------------------
        # INVERSE KINEMATICS
        # The commanded body-frame twist is transformed into per-wheel angular
        # velocities for the four omni wheels. These values are then published as a
        # joint trajectory so the robot tracks the desired planar motion.
        # ------------------------------------------------------------------
        w1 = (1/self.r_scaled) * (vx_corrected - vy_corrected - (self.L + self.l) * wz_corrected)
        w2 = (1/self.r_scaled) * (vx_corrected + vy_corrected + (self.L + self.l) * wz_corrected)
        w3 = (1/self.r_scaled) * (vx_corrected + vy_corrected - (self.L + self.l) * wz_corrected)
        w4 = (1/self.r_scaled) * (vx_corrected - vy_corrected + (self.L + self.l) * wz_corrected)

        traj_msg = JointTrajectory()
        traj_msg.joint_names = self.joint_names
        point = JointTrajectoryPoint()
        point.velocities = [w2, w1, w3, w4] 
        point.time_from_start.sec = 0; point.time_from_start.nanosec = 20000000 
        traj_msg.points = [point]
        self.wheel_pub.publish(traj_msg)

def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(OmniController())
    rclpy.shutdown()

if __name__ == '__main__':
    main()