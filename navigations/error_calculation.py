#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point  # We will use Point (x,y,z) to store (euc, lon, cross)
import math

def euler_from_quaternion(q):
    siny_cosp = 2 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return yaw

class ErrorCalculatorV2(Node):
    def __init__(self):
        super().__init__('error_calculator_v2')
        self.raw_odom = None
        self.filtered_odom = None
        self.ground_truth = None

        # ------------------------------------------------------------------
        # PERFORMANCE PUBLISHERS
        # These topics publish the decomposed motion error as Euclidean, longi-
        # tudinal, and cross-track components for both raw and filtered odometry.
        # ------------------------------------------------------------------
        self.raw_error_pub = self.create_publisher(
            Point, '/performance/raw_errors', 10)

        self.filtered_error_pub = self.create_publisher(
            Point, '/performance/filtered_errors', 10)

        self.raw_sub = self.create_subscription(
            Odometry, '/odom', self.raw_callback, 10)

        self.filtered_sub = self.create_subscription(
            Odometry, '/odometry/filtered', self.filtered_callback, 10)

        self.gt_sub = self.create_subscription(
            Odometry, '/ground_truth/pose', self.gt_callback, 10)

        self.timer = self.create_timer(0.1, self.calculate_and_publish_errors) # Faster update
        self.get_logger().info('Error Calculator V2 (with Publishers) has been started.')
        
        # ------------------------------------------------------------------
        # STATE STORAGE FOR DOWNSTREAM NODES
        # These cached values are useful to other modules, such as RL agents or
        # diagnostics nodes, that need the latest tracking error without waiting
        # for a new message from the subscribers.
        # ------------------------------------------------------------------
        self.current_raw_errors = (0.0, 0.0, 0.0)
        self.current_filtered_errors = (0.0, 0.0, 0.0)

    def raw_callback(self, msg):
        self.raw_odom = msg

    def filtered_callback(self, msg):
        self.filtered_odom = msg

    def gt_callback(self, msg):
        self.ground_truth = msg

    def calculate_all_errors(self, gt_pose, odom_pose):
        # ------------------------------------------------------------------
        # ERROR DECOMPOSITION
        # The ground-truth and odometry poses are compared in the same world frame
        # and then split into Euclidean error, longitudinal drift, and sideways
        # cross-track deviation relative to the robot's heading.
        # ------------------------------------------------------------------
        gt_x = gt_pose.position.x
        gt_y = gt_pose.position.y
        odom_x = odom_pose.position.x
        odom_y = odom_pose.position.y
        gt_yaw = euler_from_quaternion(gt_pose.orientation)
        e_x = gt_x - odom_x
        e_y = gt_y - odom_y
        euclidean_error = math.sqrt(e_x**2 + e_y**2)
        cos_yaw = math.cos(gt_yaw)
        sin_yaw = math.sin(gt_yaw)
        longitudinal_error =  e_x * cos_yaw + e_y * sin_yaw
        cross_error        = -e_x * sin_yaw + e_y * cos_yaw
        return euclidean_error, longitudinal_error, cross_error

    def calculate_and_publish_errors(self):
        if self.ground_truth is None:
            self.get_logger().warn('Waiting for ground truth data...', throttle_duration_sec=5)
            return

        gt_pose = self.ground_truth.pose.pose
        error_msg = Point()

        # ------------------------------------------------------------------
        # PUBLISH RAW AND FILTERED ERROR ESTIMATES
        # The raw topic shows the immediate odometry error, while the filtered
        # output indicates the motion quality after the estimator has corrected the
        # pose estimate.
        # ------------------------------------------------------------------
        if self.raw_odom is not None:
            euc, lon, cross = self.calculate_all_errors(gt_pose, self.raw_odom.pose.pose)
            self.current_raw_errors = (euc, lon, cross)
            error_msg.x = euc
            error_msg.y = lon
            error_msg.z = cross
            self.raw_error_pub.publish(error_msg)

        if self.filtered_odom is not None:
            euc, lon, cross = self.calculate_all_errors(gt_pose, self.filtered_odom.pose.pose)
            self.current_filtered_errors = (euc, lon, cross)
            error_msg.x = euc
            error_msg.y = lon
            error_msg.z = cross
            self.filtered_error_pub.publish(error_msg)

            # Log the filtered estimate because it is the most useful metric for
            # evaluating how well the pose fusion pipeline is tracking the robot.
            self.get_logger().info(f'Filtered Error (Euc/Lon/Cross): {euc:6.3f}m | {lon:6.3f}m | {cross:6.3f}m', throttle_duration_sec=1.0)


def main(args=None):
    rclpy.init(args=args)
    node = ErrorCalculatorV2()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()