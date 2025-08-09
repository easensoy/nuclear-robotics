#!/usr/bin/env python3

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped
from sensor_msgs.msg import Imu, LaserScan
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray
from tf2_ros import TransformBroadcaster, Buffer, TransformListener
from scipy.spatial.transform import Rotation as R

class ExtendedKalmanFilter:
    def __init__(self):
        self.n_states = 7
        self.x = np.zeros((7, 1))
        self.P = np.eye(7) * 0.1
        self.Q = np.diag([0.01, 0.01, 0.01, 0.05, 0.05, 0.05, 0.1])
        self.R = np.diag([0.1, 0.1, 0.05, 0.02, 0.02, 0.02])
        self.dt = 0.1
        
    def predict(self):
        x, y, theta, vx, vy, omega, rad = self.x.flatten()
        self.x = np.array([[x + vx*self.dt*np.cos(theta) - vy*self.dt*np.sin(theta)],
                          [y + vx*self.dt*np.sin(theta) + vy*self.dt*np.cos(theta)],
                          [theta + omega*self.dt], [vx*0.95], [vy*0.95], [omega*0.9], [rad*0.98]])
        
        F = np.eye(7)
        F[0,2] = -vx*self.dt*np.sin(theta) - vy*self.dt*np.cos(theta)
        F[0,3] = self.dt*np.cos(theta)
        F[0,4] = -self.dt*np.sin(theta)
        F[1,2] = vx*self.dt*np.cos(theta) - vy*self.dt*np.sin(theta)
        F[1,3] = self.dt*np.sin(theta)
        F[1,4] = self.dt*np.cos(theta)
        F[2,5] = self.dt
        F[3,3] = F[4,4] = 0.95
        F[5,5] = 0.9
        F[6,6] = 0.98
        
        self.P = F @ self.P @ F.T + self.Q
        
    def update(self, measurements, types):
        if not measurements: return
        
        z = np.array(measurements).reshape(-1, 1)
        H = np.zeros((len(types), 7))
        h = np.zeros((len(types), 1))
        
        for i, t in enumerate(types):
            idx = {'position_x':0, 'position_y':1, 'orientation':2, 'velocity_x':3, 
                   'velocity_y':4, 'angular_velocity':5, 'radiation':6}[t]
            H[i, idx] = 1.0
            h[i] = self.x[idx, 0]
        
        y = z - h
        S = H @ self.P @ H.T + self.R[:len(measurements), :len(measurements)]
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I_KH = np.eye(7) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R[:len(measurements), :len(measurements)] @ K.T

class NuclearRobotStateEstimator(Node):
    def __init__(self):
        super().__init__('nuclear_robot_state_estimator')
        self.ekf = ExtendedKalmanFilter()
        self.sensor_data = {}
        self.last_update_time = self.get_clock().now()
        
        self.pose_pub = self.create_publisher(PoseWithCovarianceStamped, '/robot/pose_estimated', 10)
        self.odom_pub = self.create_publisher(Odometry, '/robot/odometry_filtered', 10)
        
        self.create_subscription(Imu, '/robot/imu', self.imu_callback, 10)
        self.create_subscription(Odometry, '/robot/odom_raw', self.odometry_callback, 10)
        self.create_subscription(LaserScan, '/robot/scan', self.laser_callback, 10)
        self.create_subscription(Float64MultiArray, '/robot/radiation_sensor', self.radiation_callback, 10)
        
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)
        
        self.timer = self.create_timer(0.1, self.estimation_callback)
        
    def imu_callback(self, msg):
        r = R.from_quat([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])
        self.sensor_data['imu'] = {'orientation': r.as_euler('xyz')[2], 'angular_velocity': msg.angular_velocity.z}
        
    def odometry_callback(self, msg):
        self.sensor_data['odometry'] = {
            'position': [msg.pose.pose.position.x, msg.pose.pose.position.y],
            'velocity': [msg.twist.twist.linear.x, msg.twist.twist.linear.y]
        }
        
    def laser_callback(self, msg):
        ranges = np.array(msg.ranges)
        valid = ranges[~np.isnan(ranges) & ~np.isinf(ranges)]
        if len(valid) > 0:
            self.sensor_data['lidar'] = {'wall_distance': np.min(valid)}
    
    def radiation_callback(self, msg):
        if msg.data:
            self.sensor_data['radiation'] = {'intensity': msg.data[0]}
    
    def estimation_callback(self):
        current_time = self.get_clock().now()
        self.ekf.dt = (current_time - self.last_update_time).nanoseconds / 1e9
        self.ekf.predict()
        
        measurements, types = [], []
        
        if 'odometry' in self.sensor_data:
            measurements.extend(self.sensor_data['odometry']['position'] + self.sensor_data['odometry']['velocity'])
            types.extend(['position_x', 'position_y', 'velocity_x', 'velocity_y'])
        
        if 'imu' in self.sensor_data:
            measurements.extend([self.sensor_data['imu']['orientation'], self.sensor_data['imu']['angular_velocity']])
            types.extend(['orientation', 'angular_velocity'])
        
        if 'radiation' in self.sensor_data:
            measurements.append(self.sensor_data['radiation']['intensity'])
            types.append('radiation')
        
        if measurements:
            self.ekf.update(measurements, types)
        
        self.publish_estimated_state()
        self.last_update_time = current_time
        
    def publish_estimated_state(self):
        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = 'map'
        
        pose_msg.pose.pose.position.x = float(self.ekf.x[0, 0])
        pose_msg.pose.pose.position.y = float(self.ekf.x[1, 0])
        
        quat = R.from_euler('xyz', [0, 0, float(self.ekf.x[2, 0])]).as_quat()
        pose_msg.pose.pose.orientation.x = quat[0]
        pose_msg.pose.pose.orientation.y = quat[1]
        pose_msg.pose.pose.orientation.z = quat[2]
        pose_msg.pose.pose.orientation.w = quat[3]
        
        cov = np.zeros(36)
        cov[0] = self.ekf.P[0, 0]
        cov[7] = self.ekf.P[1, 1]
        cov[35] = self.ekf.P[2, 2]
        pose_msg.pose.covariance = cov.tolist()
        
        self.pose_pub.publish(pose_msg)
        
        odom_msg = Odometry()
        odom_msg.header = pose_msg.header
        odom_msg.child_frame_id = 'base_link'
        odom_msg.pose = pose_msg.pose
        odom_msg.twist.twist.linear.x = float(self.ekf.x[3, 0])
        odom_msg.twist.twist.linear.y = float(self.ekf.x[4, 0])
        odom_msg.twist.twist.angular.z = float(self.ekf.x[5, 0])
        
        twist_cov = np.zeros(36)
        twist_cov[0] = self.ekf.P[3, 3]
        twist_cov[7] = self.ekf.P[4, 4]
        twist_cov[35] = self.ekf.P[5, 5]
        odom_msg.twist.covariance = twist_cov.tolist()
        
        self.odom_pub.publish(odom_msg)

def main(args=None):
    rclpy.init(args=args)
    state_estimator = NuclearRobotStateEstimator()
    try:
        rclpy.spin(state_estimator)
    except KeyboardInterrupt:
        pass
    finally:
        state_estimator.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()