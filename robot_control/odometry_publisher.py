#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import math
from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster

def quaternion_from_euler(roll, pitch, yaw):
    cy = math.cos(yaw * 0.5); sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5); sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5); sr = math.sin(roll * 0.5)
    q = [0.0]*4
    q[0] = cr * cp * cy + sr * sp * sy; q[1] = sr * cp * cy - cr * sp * sy
    q[2] = cr * sp * cy + sr * cp * sy; q[3] = cr * cp * sy - sr * sp * cy
    return q

class OdometryPublisher(Node):
    def __init__(self):
        super().__init__('odometry_publisher')
        self.declare_parameter('wheel_radius', 0.05)
        self.declare_parameter('wheel_separation_x', 0.54)
        self.declare_parameter('wheel_separation_y', 0.54)
        self.declare_parameter('publish_tf', True) # Added parameter check

        self.r = float(self.get_parameter('wheel_radius').value)
        self.l = float(self.get_parameter('wheel_separation_x').value) / 2.0
        self.L = float(self.get_parameter('wheel_separation_y').value) / 2.0
        self.should_publish_tf = self.get_parameter('publish_tf').value

        self.joint_sub = self.create_subscription(JointState, 'joint_states', self.joint_state_callback, 10)
        self.odom_pub = self.create_publisher(Odometry, 'odom', 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.x = 0.0; self.y = 0.0; self.theta = 0.0
        self.last_time = self.get_clock().now()
        self.wheel_velocities = {}
        self.wheel_names = ["front_left_wheel_joint", "front_right_wheel_joint", "back_left_wheel_joint", "back_right_wheel_joint"]

        self.SQRT_2 = math.sqrt(2.0)
        
        # Debug flag to print only once
        self.has_received_joints = False
        self.get_logger().info(f'Odometry Publisher Started. Expecting joints: {self.wheel_names}')

    def joint_state_callback(self, msg: JointState):
        # Map incoming joint names to velocities
        for i, name in enumerate(msg.name):
            if name in self.wheel_names: 
                self.wheel_velocities[name] = msg.velocity[i]
        
        # Check if we have all 4 wheels
        if len(self.wheel_velocities) < 4: 
            # DEBUG: If we never find the wheels, print what we ARE seeing
            if not self.has_received_joints:
                 # Only print every 100th time to avoid spam, or just once
                 pass
            return

        if not self.has_received_joints:
            self.get_logger().info("SUCCESS: All 4 wheel joints detected! Odometry active.")
            self.has_received_joints = True

        w1 = self.wheel_velocities.get("front_right_wheel_joint", 0.0)
        w2 = self.wheel_velocities.get("front_left_wheel_joint", 0.0)
        w3 = self.wheel_velocities.get("back_left_wheel_joint", 0.0)
        w4 = self.wheel_velocities.get("back_right_wheel_joint", 0.0)

        current_time = self.get_clock().now()
        dt = (current_time - self.last_time).nanoseconds / 1.0e9
        
        if dt > 1.0: # Reset dt if lag occurs
            dt = 0.0
            
        self.last_time = current_time

        # Kinematics
        vx = (self.r * self.SQRT_2 / 4) * ( w1 + w2 + w3 + w4)
        vy = (self.r * self.SQRT_2 / 4) * (-w1 + w2 + w3 - w4)
        wz = (self.r * self.SQRT_2 / (4 * (self.L + self.l))) * (-w1 + w2 - w3 + w4)

        delta_x = (vx * math.cos(self.theta) - vy * math.sin(self.theta)) * dt
        delta_y = (vx * math.sin(self.theta) + vy * math.cos(self.theta)) * dt
        self.x += delta_x; self.y += delta_y; self.theta += wz * dt

        self.publish_odometry(current_time.to_msg(), vx, vy, wz)

    def publish_odometry(self, current_time_msg, vx, vy, wz):
        q = quaternion_from_euler(0, 0, self.theta)
        
        # 1. Publish TF (Only if allowed)
        if self.should_publish_tf:
            t = TransformStamped(); t.header.stamp = current_time_msg
            t.header.frame_id = 'odom'; t.child_frame_id = 'base_link'
            t.transform.translation.x = self.x; t.transform.translation.y = self.y
            t.transform.rotation.w = q[0]; t.transform.rotation.x = q[1]
            t.transform.rotation.y = q[2]; t.transform.rotation.z = q[3]
            self.tf_broadcaster.sendTransform(t)

        # 2. Publish Topic
        odom = Odometry(); odom.header.stamp = current_time_msg
        odom.header.frame_id = 'odom'; odom.child_frame_id = 'base_link'
        odom.pose.pose.position.x = self.x; odom.pose.pose.position.y = self.y
        odom.pose.pose.orientation.w = q[0]; odom.pose.pose.orientation.x = q[1]
        odom.pose.pose.orientation.y = q[2]; odom.pose.pose.orientation.z = q[3]
        odom.twist.twist.linear.x = vx; odom.twist.twist.linear.y = vy
        odom.twist.twist.angular.z = wz
        self.odom_pub.publish(odom)

def main(args=None):
    rclpy.init(args=args)
    node = OdometryPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()