#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import threading
from scipy.spatial import KDTree
from scipy.interpolate import griddata, RBFInterpolator
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
import json
import time

# ROS2 message imports
from std_msgs.msg import Float64MultiArray, Header
from geometry_msgs.msg import PoseStamped, Point, PointStamped
from sensor_msgs.msg import PointCloud2, PointField
from nav_msgs.msg import OccupancyGrid, Path
from visualization_msgs.msg import MarkerArray, Marker
from tf2_ros import Buffer, TransformListener, TransformException

# Custom message imports
from nuclear_robot_core.msg import RadiationField, RobotState, SafetyStatus

class RadiationMeasurement:
    """Individual radiation measurement with spatial and temporal information"""
    
    def __init__(self, position: Tuple[float, float, float], 
                 intensity: float, timestamp: float, uncertainty: float = 0.1):
        self.position = np.array(position)
        self.intensity = intensity
        self.timestamp = timestamp
        self.uncertainty = uncertainty
        self.isotope_signature = None  # Could store spectral analysis
        
class RadiationField3D:
    """3D radiation field representation with spatial interpolation"""
    
    def __init__(self, bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]):
        self.bounds = bounds  # ((x_min, x_max), (y_min, y_max), (z_min, z_max))
        self.measurements = []
        self.grid_resolution = 0.5  # meters
        self.interpolator = None
        self.gp_model = None
        self.last_update = 0.0
        
        # Create 3D grid for visualization
        self.create_interpolation_grid()
        
        # Initialize Gaussian Process for uncertainty quantification
        self.setup_gaussian_process()
    
    def create_interpolation_grid(self):
        """Create 3D interpolation grid"""
        (x_min, x_max), (y_min, y_max), (z_min, z_max) = self.bounds
        
        self.x_grid = np.arange(x_min, x_max + self.grid_resolution, self.grid_resolution)
        self.y_grid = np.arange(y_min, y_max + self.grid_resolution, self.grid_resolution)
        self.z_grid = np.arange(z_min, z_max + self.grid_resolution, self.grid_resolution)
        
        self.grid_points = np.meshgrid(self.x_grid, self.y_grid, self.z_grid, indexing='ij')
        self.interpolation_points = np.column_stack([
            self.grid_points[0].ravel(),
            self.grid_points[1].ravel(), 
            self.grid_points[2].ravel()
        ])
    
    def setup_gaussian_process(self):
        """Initialize Gaussian Process for spatial interpolation with uncertainty"""
        # Use Matern kernel for radiation field modeling (handles non-smooth behavior)
        kernel = Matern(length_scale=2.0, nu=2.5) + RBF(length_scale=1.0)
        self.gp_model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=0.1,  # Noise level
            normalize_y=True,
            n_restarts_optimizer=3
        )
    
    def add_measurement(self, measurement: RadiationMeasurement):
        """Add new radiation measurement to the field"""
        self.measurements.append(measurement)
        self.last_update = time.time()
        
        # Limit memory usage by removing old measurements
        if len(self.measurements) > 10000:
            # Keep most recent 8000 measurements
            self.measurements = self.measurements[-8000:]
    
    def update_interpolation(self):
        """Update spatial interpolation with latest measurements"""
        if len(self.measurements) < 3:
            return False
        
        # Extract positions and intensities
        positions = np.array([m.position for m in self.measurements])
        intensities = np.array([m.intensity for m in self.measurements])
        uncertainties = np.array([m.uncertainty for m in self.measurements])
        
        try:
            # Update Gaussian Process model
            self.gp_model.fit(positions, intensities)
            
            # Update RBF interpolator for fast queries
            self.interpolator = RBFInterpolator(
                positions, intensities,
                neighbors=min(50, len(positions)),
                smoothing=0.1,
                kernel='thin_plate_spline'
            )
            
            return True
        except Exception as e:
            print(f"Interpolation update failed: {e}")
            return False
    
    def query_intensity(self, position: Tuple[float, float, float]) -> Tuple[float, float]:
        """Query radiation intensity at given position with uncertainty"""
        if self.gp_model is None or len(self.measurements) < 3:
            return 0.0, 1.0  # Zero intensity, high uncertainty
        
        pos_array = np.array(position).reshape(1, -1)
        
        try:
            # Get prediction with uncertainty from Gaussian Process
            mean_pred, std_pred = self.gp_model.predict(pos_array, return_std=True)
            return float(mean_pred[0]), float(std_pred[0])
        except Exception:
            # Fallback to RBF interpolation
            if self.interpolator is not None:
                intensity = self.interpolator(pos_array)[0]
                return float(intensity), 0.5  # Moderate uncertainty
            return 0.0, 1.0
    
    def get_high_radiation_zones(self, threshold: float = 2.0) -> List[Tuple[float, float, float]]:
        """Identify zones with radiation above threshold"""
        if not self.interpolator:
            return []
        
        # Sample grid points and find high radiation areas
        grid_intensities = self.interpolator(self.interpolation_points)
        high_radiation_indices = np.where(grid_intensities > threshold)[0]
        
        return [tuple(self.interpolation_points[i]) for i in high_radiation_indices]
    
    def generate_safety_path(self, start: Tuple[float, float, float], 
                           goal: Tuple[float, float, float], 
                           max_radiation: float = 1.0) -> Optional[List[Tuple[float, float, float]]]:
        """Generate path that avoids high radiation zones"""
        if not self.interpolator:
            return None
        
        # Simple A* variant that considers radiation as cost
        # This is a simplified implementation - production would use more sophisticated path planning
        
        # Sample points along direct path and check radiation
        start_np = np.array(start)
        goal_np = np.array(goal)
        
        # Generate waypoints along path
        num_waypoints = int(np.linalg.norm(goal_np - start_np) / 0.5) + 1
        waypoints = []
        
        for i in range(num_waypoints + 1):
            alpha = i / num_waypoints
            point = start_np * (1 - alpha) + goal_np * alpha
            
            # Check radiation level at point
            intensity, _ = self.query_intensity(tuple(point))
            
            if intensity > max_radiation:
                # Try to find alternative waypoint by moving perpendicular
                for offset in [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0)]:
                    alt_point = point + np.array(offset) * 2.0
                    alt_intensity, _ = self.query_intensity(tuple(alt_point))
                    
                    if alt_intensity <= max_radiation:
                        waypoints.append(tuple(alt_point))
                        break
                else:
                    # No safe alternative found, use original point with warning
                    waypoints.append(tuple(point))
            else:
                waypoints.append(tuple(point))
        
        return waypoints

class NuclearRadiationMapper(Node):
    """Advanced radiation mapping system with real-time 3D field reconstruction"""
    
    def __init__(self):
        super().__init__('nuclear_radiation_mapper')
        
        # Initialize radiation field
        field_bounds = ((-20.0, 20.0), (-20.0, 20.0), (-2.0, 5.0))  # x, y, z bounds in meters
        self.radiation_field = RadiationField3D(field_bounds)
        
        # Sensor configuration
        self.sensor_positions = [
            (0.0, 0.0, 0.5),   # Primary sensor on robot
            (0.5, 0.0, 0.5),   # Secondary sensor - right side
            (-0.5, 0.0, 0.5),  # Secondary sensor - left side
        ]
        
        # Robot state tracking
        self.current_pose = None
        self.sensor_data_buffer = []
        self.mapping_active = False
        
        # TF2 for coordinate transformations
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        self.setup_publishers()
        self.setup_subscribers()
        
        # Mapping update timer
        self.mapping_timer = self.create_timer(0.5, self.mapping_update_callback)
        
        # Visualization timer
        self.visualization_timer = self.create_timer(2.0, self.publish_visualization)
        
        self.get_logger().info("Nuclear Radiation Mapper initialized")
    
    def setup_publishers(self):
        """Initialize ROS2 publishers"""
        # Radiation field publishing
        self.radiation_field_pub = self.create_publisher(
            RadiationField, '/radiation/field', 10)
        
        # 3D visualization for RViz
        self.radiation_markers_pub = self.create_publisher(
            MarkerArray, '/radiation/visualization', 10)
        
        # Point cloud for 3D radiation data
        self.radiation_cloud_pub = self.create_publisher(
            PointCloud2, '/radiation/point_cloud', 10)
        
        # Safe navigation paths
        self.safe_path_pub = self.create_publisher(
            Path, '/radiation/safe_path', 10)
        
        # High radiation zone alerts
        self.danger_zones_pub = self.create_publisher(
            MarkerArray, '/radiation/danger_zones', 10)
        
        # Occupancy grid for radiation-aware navigation
        self.radiation_grid_pub = self.create_publisher(
            OccupancyGrid, '/radiation/occupancy_grid', 10)
    
    def setup_subscribers(self):
        """Initialize ROS2 subscribers"""
        # Multi-sensor radiation data
        self.radiation_sensor_sub = self.create_subscription(
            Float64MultiArray, '/sensors/radiation', 
            self.radiation_sensor_callback, 10)
        
        # Robot pose for spatial correlation
        self.pose_sub = self.create_subscription(
            PoseStamped, '/robot/pose', self.pose_callback, 10)
        
        # Robot state for mission coordination
        self.robot_state_sub = self.create_subscription(
            RobotState, '/robot/state', self.robot_state_callback, 10)
        
        # Navigation goals for safe path planning
        self.nav_goal_sub = self.create_subscription(
            PoseStamped, '/move_base_simple/goal', 
            self.navigation_goal_callback, 10)
    
    def radiation_sensor_callback(self, msg):
        """Process radiation sensor measurements"""
        if self.current_pose is None:
            return
        
        timestamp = self.get_clock().now().seconds_nanoseconds()[0]
        
        # Process each sensor reading
        for i, intensity in enumerate(msg.data):
            if i < len(self.sensor_positions):
                # Transform sensor position to global coordinates
                sensor_pos_local = self.sensor_positions[i]
                sensor_pos_global = self.transform_sensor_position(sensor_pos_local)
                
                if sensor_pos_global is not None:
                    # Create radiation measurement
                    measurement = RadiationMeasurement(
                        position=sensor_pos_global,
                        intensity=intensity,
                        timestamp=timestamp,
                        uncertainty=0.1 + 0.05 * intensity  # Uncertainty increases with intensity
                    )
                    
                    # Add to radiation field
                    self.radiation_field.add_measurement(measurement)
        
        # Publish individual measurement
        self.publish_radiation_field_update()
    
    def transform_sensor_position(self, sensor_pos_local: Tuple[float, float, float]) -> Optional[Tuple[float, float, float]]:
        """Transform sensor position from robot frame to global frame"""
        if self.current_pose is None:
            return None
        
        try:
            # Get transform from robot base to map frame
            transform = self.tf_buffer.lookup_transform(
                'map', 'base_link', rclpy.time.Time())
            
            # Apply transformation (simplified - production would use full TF2 transform)
            robot_x = self.current_pose.pose.position.x
            robot_y = self.current_pose.pose.position.y
            robot_z = self.current_pose.pose.position.z
            
            # Add sensor offset (simplified transformation)
            global_x = robot_x + sensor_pos_local[0]
            global_y = robot_y + sensor_pos_local[1]
            global_z = robot_z + sensor_pos_local[2]
            
            return (global_x, global_y, global_z)
            
        except TransformException as e:
            self.get_logger().warn(f"Transform lookup failed: {e}")
            return None
    
    def pose_callback(self, msg):
        """Update robot pose for spatial correlation"""
        self.current_pose = msg
    
    def robot_state_callback(self, msg):
        """Update robot state and mapping status"""
        self.mapping_active = (msg.mission_status == "mapping" or 
                             msg.mission_status == "inspecting")
    
    def navigation_goal_callback(self, msg):
        """Generate safe path when navigation goal is received"""
        if self.current_pose is None:
            return
        
        start_pos = (
            self.current_pose.pose.position.x,
            self.current_pose.pose.position.y,
            self.current_pose.pose.position.z
        )
        
        goal_pos = (
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        )
        
        # Generate radiation-safe path
        safe_waypoints = self.radiation_field.generate_safety_path(
            start_pos, goal_pos, max_radiation=1.5)
        
        if safe_waypoints:
            self.publish_safe_path(safe_waypoints)
    
    def mapping_update_callback(self):
        """Update radiation field interpolation"""
        if not self.mapping_active:
            return
        
        # Update interpolation model
        if self.radiation_field.update_interpolation():
            self.get_logger().debug("Radiation field interpolation updated")
        
        # Publish updated radiation grid for navigation
        self.publish_radiation_occupancy_grid()
    
    def publish_radiation_field_update(self):
        """Publish current radiation field measurement"""
        if not self.radiation_field.measurements:
            return
        
        # Get latest measurement
        latest = self.radiation_field.measurements[-1]
        
        msg = RadiationField()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        
        msg.pose.position.x = latest.position[0]
        msg.pose.position.y = latest.position[1] 
        msg.pose.position.z = latest.position[2]
        
        msg.intensity = latest.intensity
        msg.uncertainty = latest.uncertainty
        msg.measurement_time = latest.timestamp
        
        self.radiation_field_pub.publish(msg)
    
    def publish_visualization(self):
        """Publish RViz visualization markers"""
        if not self.radiation_field.measurements:
            return
        
        marker_array = MarkerArray()
        
        # Radiation intensity markers
        for i, measurement in enumerate(self.radiation_field.measurements[-100:]):  # Show last 100
            marker = Marker()
            marker.header.frame_id = 'map'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'radiation_measurements'
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            
            # Position
            marker.pose.position.x = measurement.position[0]
            marker.pose.position.y = measurement.position[1]
            marker.pose.position.z = measurement.position[2]
            
            # Size based on intensity
            scale = 0.1 + min(measurement.intensity * 0.2, 1.0)
            marker.scale.x = scale
            marker.scale.y = scale
            marker.scale.z = scale
            
            # Color based on radiation level (green -> yellow -> red)
            intensity_norm = min(measurement.intensity / 5.0, 1.0)
            marker.color.r = intensity_norm
            marker.color.g = 1.0 - intensity_norm * 0.5
            marker.color.b = 0.0
            marker.color.a = 0.7
            
            marker_array.markers.append(marker)
        
        # High radiation zone warnings
        danger_zones = self.radiation_field.get_high_radiation_zones(threshold=2.0)
        for i, zone_pos in enumerate(danger_zones[:20]):  # Limit to 20 zones
            marker = Marker()
            marker.header.frame_id = 'map'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'danger_zones'
            marker.id = i + 1000
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            
            marker.pose.position.x = zone_pos[0]
            marker.pose.position.y = zone_pos[1]
            marker.pose.position.z = zone_pos[2]
            
            marker.scale.x = 1.0
            marker.scale.y = 1.0
            marker.scale.z = 0.5
            
            # Bright red for danger zones
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 0.5
            
            marker_array.markers.append(marker)
        
        self.radiation_markers_pub.publish(marker_array)
        self.danger_zones_pub.publish(marker_array)
    
    def publish_radiation_occupancy_grid(self):
        """Publish radiation data as occupancy grid for Nav2 integration"""
        if len(self.radiation_field.measurements) < 10:
            return
        
        # Create occupancy grid
        grid = OccupancyGrid()
        grid.header.frame_id = 'map'
        grid.header.stamp = self.get_clock().now().to_msg()
        
        # Grid parameters
        resolution = 0.5  # 0.5m per cell
        width = int((self.radiation_field.bounds[0][1] - self.radiation_field.bounds[0][0]) / resolution)
        height = int((self.radiation_field.bounds[1][1] - self.radiation_field.bounds[1][0]) / resolution)
        
        grid.info.resolution = resolution
        grid.info.width = width
        grid.info.height = height
        grid.info.origin.position.x = self.radiation_field.bounds[0][0]
        grid.info.origin.position.y = self.radiation_field.bounds[1][0]
        grid.info.origin.position.z = 0.0
        
        # Fill grid with radiation-based occupancy values
        grid_data = []
        for y in range(height):
            for x in range(width):
                world_x = grid.info.origin.position.x + x * resolution
                world_y = grid.info.origin.position.y + y * resolution
                world_z = 0.0  # Ground level
                
                intensity, uncertainty = self.radiation_field.query_intensity((world_x, world_y, world_z))
                
                # Convert radiation intensity to occupancy value
                # 0-50: safe (low radiation)
                # 51-80: caution (medium radiation)  
                # 81-100: danger (high radiation)
                if intensity < 0.5:
                    occupancy = 0  # Free space
                elif intensity < 2.0:
                    occupancy = int(50 + intensity * 15)  # Caution zone
                else:
                    occupancy = min(100, int(80 + intensity * 5))  # Danger zone
                
                grid_data.append(occupancy)
        
        grid.data = grid_data
        self.radiation_grid_pub.publish(grid)
    
    def publish_safe_path(self, waypoints: List[Tuple[float, float, float]]):
        """Publish radiation-safe navigation path"""
        path = Path()
        path.header.frame_id = 'map'
        path.header.stamp = self.get_clock().now().to_msg()
        
        for waypoint in waypoints:
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = waypoint[0]
            pose.pose.position.y = waypoint[1]
            pose.pose.position.z = waypoint[2]
            pose.pose.orientation.w = 1.0
            
            path.poses.append(pose)
        
        self.safe_path_pub.publish(path)
        self.get_logger().info(f"Published safe path with {len(waypoints)} waypoints")

def main(args=None):
    rclpy.init(args=args)
    
    radiation_mapper = NuclearRadiationMapper()
    
    try:
        rclpy.spin(radiation_mapper)
    except KeyboardInterrupt:
        radiation_mapper.get_logger().info("Radiation mapper shutting down")
    finally:
        radiation_mapper.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()