#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from robot_localization.srv import SetPose
import time
import csv
import math
from datetime import datetime

class MotionTestOpenLoop(Node):
    def __init__(self):
        super().__init__('motion_test_openloop')
        
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.reset_sim = self.create_client(Empty, '/reset_simulation')
        self.reset_ekf = self.create_client(SetPose, '/set_pose')
        self.odom_sub = self.create_subscription(Odometry, '/ground_truth/pose', self.odom_cb, 10)
        
        self.pose = None
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = f"Results_NoController_Blind_{timestamp}.csv"
        self.csv_file = open(self.filename, 'w', newline='')
        self.writer = csv.writer(self.csv_file)
        self.writer.writerow(['Iteration', 'Time', 'Controller', 'Test_Name', 'Step', 'Goal_X', 'Goal_Y', 'Robot_X', 'Robot_Y', 'Robot_Yaw', 'Euclidean_Err', 'Long_Err', 'Cross_Err'])
        
        v = 1.0
        d = 3.5355 # 5.0 / sqrt(2)
        
        self.tests = [
            {'name': '1_Forward',   'vx': v,  'vy': 0.0, 'goal': [5.0, 0.0]},
            {'name': '2_Backward',  'vx': -v, 'vy': 0.0, 'goal': [-5.0, 0.0]},
            {'name': '3_Left',      'vx': 0.0, 'vy': v,  'goal': [0.0, 5.0]},
            {'name': '4_Right',     'vx': 0.0, 'vy': -v, 'goal': [0.0, -5.0]},
            {'name': '5_Diag_FL',   'vx': 0.707,  'vy': 0.707,   'goal': [d, d]}, 
            {'name': '6_Diag_FR',   'vx': 0.707,  'vy': -0.707,  'goal': [d, -d]},
            {'name': '7_Diag_BL',   'vx': -0.707, 'vy': 0.707,   'goal': [-d, d]},
            {'name': '8_Diag_BR',   'vx': -0.707, 'vy': -0.707,  'goal': [-d, -d]},
        ]
        
        self.current_iteration = 0
        self.max_iterations = 5 
        self.current_test_idx = 0
        self.state = "INIT"; self.start_time = 0
        self.timer = self.create_timer(0.05, self.control_loop) 

    def odom_cb(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y); cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.pose = [p.x, p.y, math.atan2(siny, cosy)]

    def get_errors(self, goal):
        if not self.pose: return 0, 0, 0
        dx = goal[0] - self.pose[0]; dy = goal[1] - self.pose[1]
        dist = math.sqrt(dx**2 + dy**2)
        
        g_dist = math.sqrt(goal[0]**2 + goal[1]**2)
        if g_dist < 1e-3: return dist, 0, 0
        
        ux = goal[0] / g_dist; uy = goal[1] / g_dist
        projection = self.pose[0] * ux + self.pose[1] * uy
        lon = g_dist - projection
        cross = -self.pose[0] * uy + self.pose[1] * ux
        return dist, lon, cross

    def control_loop(self):
        if not self.pose: return
        now = time.time()
        
        if self.state == "INIT":
            self.state = "RESET"
            if self.reset_sim.service_is_ready(): self.reset_sim.call_async(Empty.Request())
            if self.reset_ekf.service_is_ready(): 
                req = SetPose.Request(); req.pose.header.frame_id='odom'; req.pose.pose.pose.orientation.w=1.0
                self.reset_ekf.call_async(req)
            self.start_time = now + 1.0 
            
        elif self.state == "RESET":
            self.cmd_pub.publish(Twist())
            if self.reset_sim.service_is_ready(): self.reset_sim.call_async(Empty.Request())
            if now > self.start_time:
                self.state = "RUN"
                self.start_time = now
                print(f"ITERATION {self.current_iteration+1}/5 | Test: {self.tests[self.current_test_idx]['name']}")

        elif self.state == "RUN":
            dt = now - self.start_time
            test = self.tests[self.current_test_idx]
            
            # --- BLIND CONTROL LOGIC ---
            # Drive for exactly 5.0 seconds (5m @ 1m/s)
            # Do NOT check sensors for early stop.
            cmd = Twist()
            if dt < 5.0:
                cmd.linear.x = float(test['vx']); cmd.linear.y = float(test['vy'])
            else:
                # After 5s, stop motors. 
                # Friction will eventually stop the robot.
                cmd.linear.x = 0.0; cmd.linear.y = 0.0
                
            self.cmd_pub.publish(cmd)
            
            # Log Data
            dist, lon, cross = self.get_errors(test['goal'])
            self.writer.writerow([self.current_iteration, dt, "No_Controller", test['name'], int(dt*20), 
                                  test['goal'][0], test['goal'][1], self.pose[0], self.pose[1], self.pose[2], 
                                  dist, lon, cross])
            
            # --- TIMEOUT ONLY ---
            # We record for 15s total to see where it settles after stopping
            if dt >= 15.0:
                print(f"FINISHED: {test['name']} (Blind Run)")
                self.advance_test(now)

        elif self.state == "DONE":
            self.cmd_pub.publish(Twist())
            self.csv_file.close()
            print("No-Controller 5-Iteration Test Complete.")
            self.destroy_node()
            rclpy.shutdown()

    def advance_test(self, now):
        self.current_test_idx += 1
        if self.current_test_idx >= len(self.tests):
            self.current_iteration += 1
            self.current_test_idx = 0
            if self.current_iteration >= self.max_iterations:
                self.state = "DONE"
            else:
                self.state = "RESET"; self.start_time = now
        else:
            self.state = "RESET"; self.start_time = now

def main():
    rclpy.init()
    rclpy.spin(MotionTestOpenLoop())

if __name__ == '__main__':
    main()