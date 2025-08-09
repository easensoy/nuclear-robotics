#!/usr/bin/env python3

"""
Advanced Extended Kalman Filter for Nuclear Radiation Mapping Robot
Demonstrates modern state estimation techniques for Createc-style applications

This implementation provides multi-sensor fusion for autonomous navigation
in nuclear environments, combining odometry, IMU, LIDAR, and radiation sensors.
"""

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped, TwistStamped
from sensor_msgs.msg import Imu, LaserScan, PointCloud2
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray
import tf2_py
import tf2_geometry_msgs
from tf2_ros import TransformBroadcaster, Buffer, TransformListener
import math
from scipy.spatial.transform import Rotation as R

class ExtendedKalmanFilter:
    """
    Advanced EKF implementation for nuclear robot state estimation
    
    State vector: [x, y, theta, vx, vy, omega, radiation_intensity]
    Handles sensor fusion from multiple modalities including radiation sensors
    """
    
    def __init__(self):
        # State dimension (7D: pose + velocity + radiation)
        self.n_states = 7
        self.n_observations = 6  # Variable based on available sensors
        
        # State vector: [x, y, theta, vx, vy, omega, radiation]
        self.x = np.zeros((self.n_states, 1))
        
        # State covariance matrix
        self.P = np.eye(self.n_states) * 0.1
        
        # Process noise covariance
        self.Q = np.diag([0.01, 0.01, 0.01, 0.05, 0.05, 0.05, 0.1])
        
        # Measurement noise covariance (dynamic based on sensors)
        self.R = np.diag([0.1, 0.1, 0.05, 0.02, 0.02, 0.02])
        
        self.dt = 0.1  # Time step
        
    def predict(self, control_input=None):
        """
        Prediction step using kinematic motion model
        Incorporates control inputs for more accurate prediction
        """
        # Extract state components
        x, y, theta, vx, vy, omega, radiation = self.x.flatten()
        
        # Kinematic motion model
        x_pred = x + vx * self.dt * np.cos(theta) - vy * self.dt * np.sin(theta)
        y_pred = y + vx * self.dt * np.sin(theta) + vy * self.dt * np.cos(theta)
        theta_pred = theta + omega * self.dt
        
        # Velocity prediction (with damping)
        vx_pred = vx * 0.95  # Velocity damping
        vy_pred = vy * 0.95
        omega_pred = omega * 0.9
        
        # Radiation intensity prediction (with decay)
        radiation_pred = radiation * 0.98
        
        # Update predicted state
        self.x = np.array([[x_pred], [y_pred], [theta_pred], 
                          [vx_pred], [vy_pred], [omega_pred], [radiation_pred]])
        
        # Jacobian of motion model
        F = self.compute_motion_jacobian()
        
        # Predict covariance
        self.P = F @ self.P @ F.T + self.Q
        
    def compute_motion_jacobian(self):
        """
        Compute Jacobian of motion model for linearization
        Critical for EKF accuracy in nonlinear systems
        """
        x, y, theta, vx, vy, omega, radiation = self.x.flatten()
        
        F = np.eye(self.n_states)
        
        # Partial derivatives for position updates
        F[0, 2] = -vx * self.dt * np.sin(theta) - vy * self.dt * np.cos(theta)  # dx/dtheta
        F[0, 3] = self.dt * np.cos(theta)  # dx/dvx
        F[0, 4] = -self.dt * np.sin(theta)  # dx/dvy
        
        F[1, 2] = vx * self.dt * np.cos(theta) - vy * self.dt * np.sin(theta)  # dy/dtheta
        F[1, 3] = self.dt * np.sin(theta)  # dy/dvx
        F[1, 4] = self.dt * np.cos(theta)  # dy/dvy
        
        F[2, 5] = self.dt  # dtheta/domega
        
        # Velocity damping terms
        F[3, 3] = 0.95
        F[4, 4] = 0.95
        F[5, 5] = 0.9
        
        # Radiation decay
        F[6, 6] = 0.98
        
        return F
    
    def update(self, measurements, measurement_types):
        """
        Update step with multi-sensor measurements
        Handles variable sensor availability for robust operation
        """
        if len(measurements) == 0:
            return
            
        # Build measurement vector and observation matrix
        z = np.array(measurements).reshape(-1, 1)
        H = self.compute_observation_jacobian(measurement_types)
        
        # Predicted measurements
        h = self.compute_predicted_measurements(measurement_types)
        
        # Innovation
        y = z - h
        
        # Innovation covariance
        S = H @ self.P @ H.T + self.R[:len(measurements), :len(measurements)]
        
        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # State update
        self.x = self.x + K @ y
        
        # Covariance update (Joseph form for numerical stability)
        I_KH = np.eye(self.n_states) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R[:len(measurements), :len(measurements)] @ K.T
        
    def compute_observation_jacobian(self, measurement_types):
        """
        Compute observation Jacobian for available sensors
        Adapts to sensor availability for robust operation
        """
        H = np.zeros((len(measurement_types), self.n_states))
        
        for i, sensor_type in enumerate(measurement_types):
            if sensor_type == 'position_x':
                H[i, 0] = 1.0
            elif sensor_type == 'position_y':
                H[i, 1] = 1.0
            elif sensor_type == 'orientation':
                H[i, 2] = 1.0
            elif sensor_type == 'velocity_x':
                H[i, 3] = 1.0
            elif sensor_type == 'velocity_y':
                H[i, 4] = 1.0
            elif sensor_type == 'angular_velocity':
                H[i, 5] = 1.0
            elif sensor_type == 'radiation':
                H[i, 6] = 1.0
                
        return H
    
    def compute_predicted_measurements(self, measurement_types):
        """
        Compute predicted measurements based on current state
        """
        h = np.zeros((len(measurement_types), 1))
        
        for i, sensor_type in enumerate(measurement_types):
            if sensor_type == 'position_x':
                h[i] = self.x[0, 0]
            elif sensor_type == 'position_y':
                h[i] = self.x[1, 0]
            elif sensor_type == 'orientation':
                h[i] = self.x[2, 0]
            elif sensor_type == 'velocity_x':
                h[i] = self.x[3, 0]
            elif sensor_type == 'velocity_y':
                h[i] = self.x[4, 0]
            elif sensor_type == 'angular_velocity':
                h[i] = self.x[5, 0]
            elif sensor_type == 'radiation':
                h[i] = self.x[6, 0]
                
        return h

class NuclearRobotStateEstimator(Node):
    """
    ROS2 node implementing advanced state estimation for nuclear robots
    Integrates multiple sensors for robust navigation in hazardous environments
    """
    
    def __init__(self):
        super().__init__('nuclear_robot_state_estimator')
        
        # Initialize Extended Kalman Filter
        self.ekf = ExtendedKalmanFilter()
        
        # Publishers
        self.pose_pub = self.create_publisher(
            PoseWithCovarianceStamped, 
            '/robot/pose_estimated', 
            10
        )
        
        self.odom_pub = self.create_publisher(
            Odometry,
            '/robot/odometry_filtered',
            10
        )
        
        # Subscribers for multi-sensor input
        self.imu_sub = self.create_subscription(
            Imu,
            '/robot/imu',
            self.imu_callback,
            10
        )
        
        self.odom_sub = self.create_subscription(
            Odometry,
            '/robot/odom_raw',
            self.odometry_callback,
            10
        )
        
        self.laser_sub = self.create_subscription(
            LaserScan,
            '/robot/scan',
            self.laser_callback,
            10
        )
        
        self.radiation_sub = self.create_subscription(
            Float64MultiArray,
            '/robot/radiation_sensor',
            self.radiation_callback,
            10
        )
        
        # Transform handling
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # Timing and sensor data storage
        self.last_update_time = self.get_clock().now()
        self.sensor_data = {}
        
        # Main estimation timer
        self.timer = self.create_timer(0.1, self.estimation_callback)
        
        self.get_logger().info('Nuclear Robot State Estimator initialized')
        
    def imu_callback(self, msg):
        """Process IMU measurements for orientation and angular velocity"""
        # Extract orientation (convert quaternion to euler)
        orientation_q = msg.orientation
        r = R.from_quat([orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w])
        roll, pitch, yaw = r.as_euler('xyz')
        
        # Store IMU data
        self.sensor_data['imu'] = {
            'orientation': yaw,
            'angular_velocity': msg.angular_velocity.z,
            'linear_acceleration': [msg.linear_acceleration.x, msg.linear_acceleration.y],
            'timestamp': self.get_clock().now()
        }
        
    def odometry_callback(self, msg):
        """Process odometry measurements for position and velocity"""
        self.sensor_data['odometry'] = {
            'position': [msg.pose.pose.position.x, msg.pose.pose.position.y],
            'velocity': [msg.twist.twist.linear.x, msg.twist.twist.linear.y],
            'timestamp': self.get_clock().now()
        }
        
    def laser_callback(self, msg):
        """Process LIDAR data for position correction using landmark detection"""
        # Simplified landmark detection for demonstration
        # In real implementation, would use SLAM or feature matching
        ranges = np.array(msg.ranges)
        valid_ranges = ranges[~np.isnan(ranges) & ~np.isinf(ranges)]
        
        if len(valid_ranges) > 0:
            # Simple wall detection for position correction
            min_range = np.min(valid_ranges)
            self.sensor_data['lidar'] = {
                'wall_distance': min_range,
                'range_mean': np.mean(valid_ranges),
                'timestamp': self.get_clock().now()
            }
        
    def radiation_callback(self, msg):
        """Process radiation sensor measurements"""
        if len(msg.data) > 0:
            self.sensor_data['radiation'] = {
                'intensity': msg.data[0],
                'gradient': msg.data[1:4] if len(msg.data) > 3 else [0, 0, 0],
                'timestamp': self.get_clock().now()
            }
    
    def estimation_callback(self):
        """Main estimation loop - prediction and update steps"""
        current_time = self.get_clock().now()
        dt = (current_time - self.last_update_time).nanoseconds / 1e9
        self.ekf.dt = dt
        
        # Prediction step
        self.ekf.predict()
        
        # Collect available measurements
        measurements = []
        measurement_types = []
        
        # Add odometry measurements if available
        if 'odometry' in self.sensor_data:
            odom_data = self.sensor_data['odometry']
            measurements.extend(odom_data['position'])
            measurements.extend(odom_data['velocity'])
            measurement_types.extend(['position_x', 'position_y', 'velocity_x', 'velocity_y'])
        
        # Add IMU measurements if available
        if 'imu' in self.sensor_data:
            imu_data = self.sensor_data['imu']
            measurements.append(imu_data['orientation'])
            measurements.append(imu_data['angular_velocity'])
            measurement_types.extend(['orientation', 'angular_velocity'])
        
        # Add radiation measurements if available
        if 'radiation' in self.sensor_data:
            radiation_data = self.sensor_data['radiation']
            measurements.append(radiation_data['intensity'])
            measurement_types.append('radiation')
        
        # Update step with available measurements
        if measurements:
            self.ekf.update(measurements, measurement_types)
        
        # Publish estimated state
        self.publish_estimated_state()
        
        self.last_update_time = current_time
        
    def publish_estimated_state(self):
        """Publish the estimated robot state"""
        # Publish pose with covariance
        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = 'map'
        
        # Position
        pose_msg.pose.pose.position.x = float(self.ekf.x[0, 0])
        pose_msg.pose.pose.position.y = float(self.ekf.x[1, 0])
        pose_msg.pose.pose.position.z = 0.0
        
        # Orientation (convert yaw to quaternion)
        yaw = float(self.ekf.x[2, 0])
        r = R.from_euler('xyz', [0, 0, yaw])
        quat = r.as_quat()
        pose_msg.pose.pose.orientation.x = quat[0]
        pose_msg.pose.pose.orientation.y = quat[1]
        pose_msg.pose.pose.orientation.z = quat[2]
        pose_msg.pose.pose.orientation.w = quat[3]
        
        # Covariance (6x6 for pose)
        pose_covariance = np.zeros(36)
        pose_covariance[0] = self.ekf.P[0, 0]   # x
        pose_covariance[7] = self.ekf.P[1, 1]   # y
        pose_covariance[35] = self.ekf.P[2, 2]  # yaw
        pose_msg.pose.covariance = pose_covariance.tolist()
        
        self.pose_pub.publish(pose_msg)
        
        # Publish full odometry
        odom_msg = Odometry()
        odom_msg.header = pose_msg.header
        odom_msg.child_frame_id = 'base_link'
        odom_msg.pose = pose_msg.pose
        
        # Velocity
        odom_msg.twist.twist.linear.x = float(self.ekf.x[3, 0])
        odom_msg.twist.twist.linear.y = float(self.ekf.x[4, 0])
        odom_msg.twist.twist.angular.z = float(self.ekf.x[5, 0])
        
        # Velocity covariance
        twist_covariance = np.zeros(36)
        twist_covariance[0] = self.ekf.P[3, 3]   # vx
        twist_covariance[7] = self.ekf.P[4, 4]   # vy
        twist_covariance[35] = self.ekf.P[5, 5]  # omega
        odom_msg.twist.covariance = twist_covariance.tolist()
        
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