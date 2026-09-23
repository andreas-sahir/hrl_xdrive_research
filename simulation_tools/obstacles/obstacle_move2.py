#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from geometry_msgs.msg import Twist
import math

class ObstacleMover(Node):
    def __init__(self):
        super().__init__('obstacle_mover')
        
        # CRITICAL: Sync to Gazebo's /clock so motion scales with sim speed
        self.set_parameters([Parameter('use_sim_time', Parameter.Type.BOOL, True)])
        
        self.pubs = []
        for i in range(1, 5):
            pub = self.create_publisher(Twist, f'/obs{i}/cmd_vel', 10)
            self.pubs.append(pub)
            
        self.timer = self.create_timer(0.05, self.timer_callback)
        self.start_time = self.get_clock().now()
        
        self.travel_distance = 3.0
        self.cycle_time = 7.5       
        self.get_logger().info(f"Dynamic Obstacles Patrolling [Sim-Time Mode].")

    def timer_callback(self):
        now = self.get_clock().now()
        t = (now - self.start_time).nanoseconds / 1e9
        
        amplitude = self.travel_distance / 2.0 
        omega = -(2 * math.pi) / self.cycle_time
        
        msg = Twist()
        msg.linear.y = amplitude * omega * math.cos(omega * t)
        
        for pub in self.pubs:
            pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = ObstacleMover()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Stopping obstacles...")
    finally:
        stop_msg = Twist()
        for pub in node.pubs:
            pub.publish(stop_msg)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()