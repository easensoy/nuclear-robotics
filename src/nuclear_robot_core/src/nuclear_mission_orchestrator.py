#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
import numpy as np
import threading
import yaml
import json
from enum import Enum
from typing import List, Dict, Optional, Tuple

# ROS2 message imports
from std_msgs.msg import String, Bool, Float64MultiArray
from geometry_msgs.msg import PoseStamped, Twist, PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid, Path
from sensor_msgs.msg import PointCloud2, LaserScan
from visualization_msgs.msg import MarkerArray, Marker
from diagnostic_msgs.msg import DiagnosticArray

# Navigation and planning imports
from nav2_msgs.action import NavigateToPose, FollowWaypoints
from nav2_msgs.srv import ManageLifecycleNodes, LoadMap
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import PlanningScene, RobotTrajectory
from control_msgs.action import FollowJointTrajectory

# PX4 drone integration
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand

# Custom message imports
from nuclear_robot_core.msg import RadiationField, RobotState, SafetyStatus
from nuclear_robot_core.srv import EmergencyStop, SetMission

class MissionState(Enum):
    IDLE = "idle"
    PLANNING = "planning"
    NAVIGATING = "navigating"
    MAPPING = "mapping"
    INSPECTING = "inspecting"
    EMERGENCY_STOP = "emergency_stop"
    RETURNING_HOME = "returning_home"
    MISSION_COMPLETE = "mission_complete"

class RobotType(Enum):
    GROUND_ROVER = "ground_rover"
    AERIAL_DRONE = "aerial_drone"
    MANIPULATOR_ARM = "manipulator_arm"

class PDDLPlanner:
    """High-level task planning using PDDL (Planning Domain Definition Language)"""
    
    def __init__(self):
        self.domain_file = None
        self.problem_file = None
        self.current_plan = []
        
    def load_domain(self, domain_path: str):
        """Load PDDL domain definition for nuclear facility operations"""
        try:
            with open(domain_path, 'r') as f:
                self.domain_file = f.read()
            return True
        except Exception as e:
            print(f"Error loading PDDL domain: {e}")
            return False
    
    def generate_inspection_plan(self, areas: List[str], radiation_levels: Dict[str, float]) -> List[Dict]:
        """Generate high-level inspection plan based on radiation data"""
        plan = []
        
        # Sort areas by radiation level (lowest first for safety)
        sorted_areas = sorted(areas, key=lambda x: radiation_levels.get(x, 0.0))
        
        for area in sorted_areas:
            rad_level = radiation_levels.get(area, 0.0)
            
            if rad_level < 1.0:  # Low radiation - detailed inspection
                plan.append({
                    'action': 'detailed_inspection',
                    'location': area,
                    'duration': 300,  # 5 minutes
                    'robot_type': RobotType.GROUND_ROVER
                })
            elif rad_level < 5.0:  # Medium radiation - aerial survey
                plan.append({
                    'action': 'aerial_survey',
                    'location': area,
                    'duration': 180,  # 3 minutes
                    'robot_type': RobotType.AERIAL_DRONE
                })
            else:  # High radiation - minimal exposure mapping
                plan.append({
                    'action': 'quick_mapping',
                    'location': area,
                    'duration': 60,   # 1 minute
                    'robot_type': RobotType.AERIAL_DRONE
                })
        
        return plan

class NuclearMissionOrchestrator(Node):
    """Central coordination system for nuclear robotics operations"""
    
    def __init__(self):
        super().__init__('nuclear_mission_orchestrator')
        
        # System state
        self.mission_state = MissionState.IDLE
        self.current_robot_type = RobotType.GROUND_ROVER
        self.mission_config = {}
        self.safety_override = False
        self.emergency_stop_active = False
        
        # Planning components
        self.pddl_planner = PDDLPlanner()
        self.current_plan = []
        self.plan_index = 0
        
        # Callback groups for concurrent operations
        self.navigation_group = ReentrantCallbackGroup()
        self.manipulation_group = ReentrantCallbackGroup()
        self.safety_group = ReentrantCallbackGroup()
        
        self.setup_publishers()
        self.setup_subscribers()
        self.setup_action_clients()
        self.setup_services()
        
        # Initialize Isaac Lab integration
        self.setup_isaac_lab_integration()
        
        # Load mission parameters
        self.load_mission_config()
        
        # Main orchestration timer
        self.orchestration_timer = self.create_timer(0.1, self.orchestration_callback)
        
        self.get_logger().info("Nuclear Mission Orchestrator initialized")
    
    def setup_publishers(self):
        """Initialize ROS2 publishers"""
        self.mission_status_pub = self.create_publisher(String, '/mission/status', 10)
        self.robot_commands_pub = self.create_publisher(Twist, '/robot/cmd_vel', 10)
        self.visualization_pub = self.create_publisher(MarkerArray, '/mission/visualization', 10)
        self.safety_alert_pub = self.create_publisher(String, '/safety/alerts', 10)
        
        # PX4 drone control publishers
        self.offboard_control_pub = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', 10)
        self.trajectory_setpoint_pub = self.create_publisher(
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint', 10)
        self.vehicle_command_pub = self.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', 10)
    
    def setup_subscribers(self):
        """Initialize ROS2 subscribers"""
        self.robot_state_sub = self.create_subscription(
            RobotState, '/robot/state', self.robot_state_callback, 10)
        self.radiation_sub = self.create_subscription(
            RadiationField, '/radiation/field', self.radiation_field_callback, 10)
        self.safety_status_sub = self.create_subscription(
            SafetyStatus, '/safety/status', self.safety_status_callback, 10, 
            callback_group=self.safety_group)
        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped, '/robot/pose_estimated', 
            self.pose_callback, 10)
        
        # Navigation feedback
        self.map_sub = self.create_subscription(
            OccupancyGrid, '/map', self.map_callback, 10)
    
    def setup_action_clients(self):
        """Initialize action clients for navigation and manipulation"""
        # Nav2 navigation
        self.navigate_to_pose_client = ActionClient(
            self, NavigateToPose, 'navigate_to_pose', 
            callback_group=self.navigation_group)
        self.follow_waypoints_client = ActionClient(
            self, FollowWaypoints, 'follow_waypoints',
            callback_group=self.navigation_group)
        
        # MoveIt manipulation
        self.move_group_client = ActionClient(
            self, MoveGroup, 'move_action',
            callback_group=self.manipulation_group)
        
        # Joint trajectory control
        self.joint_trajectory_client = ActionClient(
            self, FollowJointTrajectory, 'joint_trajectory_controller/follow_joint_trajectory',
            callback_group=self.manipulation_group)
    
    def setup_services(self):
        """Initialize service clients and servers"""
        # Emergency stop service
        self.emergency_stop_service = self.create_service(
            EmergencyStop, '/emergency_stop', self.emergency_stop_callback,
            callback_group=self.safety_group)
        
        # Mission configuration service
        self.set_mission_service = self.create_service(
            SetMission, '/set_mission', self.set_mission_callback)
        
        # Nav2 lifecycle management
        self.lifecycle_client = self.create_client(
            ManageLifecycleNodes, '/lifecycle_manager/manage_nodes')
    
    def setup_isaac_lab_integration(self):
        """Initialize Isaac Lab simulation integration"""
        try:
            # Isaac Lab specific setup will be added here
            # This would typically involve connecting to Isaac Lab's Python API
            self.isaac_lab_connected = False
            self.get_logger().info("Isaac Lab integration initialized")
        except Exception as e:
            self.get_logger().error(f"Isaac Lab initialization failed: {e}")
            self.isaac_lab_connected = False
    
    def load_mission_config(self):
        """Load mission configuration parameters"""
        try:
            # This would typically load from a YAML configuration file
            self.mission_config = {
                'max_radiation_exposure': 10.0,  # mSv/h
                'safety_timeout': 300.0,  # seconds
                'inspection_areas': [
                    'reactor_hall', 'cooling_system', 'waste_storage', 
                    'decontamination_area', 'control_room'
                ],
                'waypoint_tolerance': 0.5,  # meters
                'teb_planner_config': {
                    'max_vel_x': 0.8,
                    'max_vel_theta': 1.0,
                    'acc_lim_x': 2.0,
                    'acc_lim_theta': 3.0,
                    'min_obstacle_dist': 0.5,
                    'inflation_dist': 0.6
                }
            }
            self.get_logger().info("Mission configuration loaded")
        except Exception as e:
            self.get_logger().error(f"Failed to load mission config: {e}")
    
    def orchestration_callback(self):
        """Main orchestration loop - coordinates all system components"""
        try:
            # Check safety status first
            if self.emergency_stop_active:
                self.handle_emergency_stop()
                return
            
            # State machine execution
            if self.mission_state == MissionState.IDLE:
                self.handle_idle_state()
            elif self.mission_state == MissionState.PLANNING:
                self.handle_planning_state()
            elif self.mission_state == MissionState.NAVIGATING:
                self.handle_navigation_state()
            elif self.mission_state == MissionState.MAPPING:
                self.handle_mapping_state()
            elif self.mission_state == MissionState.INSPECTING:
                self.handle_inspection_state()
            elif self.mission_state == MissionState.RETURNING_HOME:
                self.handle_return_home_state()
            
            # Publish current mission status
            self.publish_mission_status()
            
        except Exception as e:
            self.get_logger().error(f"Orchestration error: {e}")
            self.transition_to_emergency_stop()
    
    def handle_planning_state(self):
        """Execute high-level PDDL planning for mission objectives"""
        if not self.current_plan:
            # Generate new plan using PDDL planner
            areas = self.mission_config.get('inspection_areas', [])
            radiation_data = getattr(self, 'latest_radiation_data', {})
            
            self.current_plan = self.pddl_planner.generate_inspection_plan(
                areas, radiation_data)
            
            if self.current_plan:
                self.plan_index = 0
                self.mission_state = MissionState.NAVIGATING
                self.get_logger().info(f"Generated plan with {len(self.current_plan)} tasks")
            else:
                self.get_logger().error("Failed to generate mission plan")
                self.mission_state = MissionState.IDLE
    
    def handle_navigation_state(self):
        """Execute TEB planner enhanced navigation"""
        if self.plan_index < len(self.current_plan):
            current_task = self.current_plan[self.plan_index]
            
            # Switch robot type if needed
            if current_task['robot_type'] != self.current_robot_type:
                self.switch_robot_type(current_task['robot_type'])
            
            # Execute navigation based on robot type
            if self.current_robot_type == RobotType.GROUND_ROVER:
                self.execute_ground_navigation(current_task)
            elif self.current_robot_type == RobotType.AERIAL_DRONE:
                self.execute_aerial_navigation(current_task)
            
        else:
            self.mission_state = MissionState.MISSION_COMPLETE
    
    def execute_ground_navigation(self, task: Dict):
        """Execute ground robot navigation using Nav2 with TEB planner"""
        target_location = task['location']
        
        # Create navigation goal
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        
        # Set target pose based on location (this would be loaded from config)
        location_poses = {
            'reactor_hall': (10.0, 5.0, 0.0),
            'cooling_system': (15.0, 10.0, 1.57),
            'waste_storage': (5.0, 15.0, 3.14),
            'decontamination_area': (20.0, 8.0, -1.57),
            'control_room': (0.0, 0.0, 0.0)
        }
        
        if target_location in location_poses:
            x, y, yaw = location_poses[target_location]
            goal_msg.pose.pose.position.x = x
            goal_msg.pose.pose.position.y = y
            goal_msg.pose.pose.orientation.z = np.sin(yaw / 2.0)
            goal_msg.pose.pose.orientation.w = np.cos(yaw / 2.0)
            
            # Send navigation goal
            self.navigate_to_pose_client.send_goal_async(
                goal_msg, feedback_callback=self.navigation_feedback_callback)
            
            self.mission_state = MissionState.MAPPING
    
    def execute_aerial_navigation(self, task: Dict):
        """Execute drone navigation using PX4 integration"""
        # PX4 offboard control mode
        offboard_msg = OffboardControlMode()
        offboard_msg.position = True
        offboard_msg.velocity = False
        offboard_msg.acceleration = False
        offboard_msg.attitude = False
        offboard_msg.body_rate = False
        offboard_msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_control_pub.publish(offboard_msg)
        
        # Set trajectory setpoint
        trajectory_msg = TrajectorySetpoint()
        
        # Set target position based on location
        location_positions = {
            'reactor_hall': (10.0, 5.0, -3.0),  # 3m altitude
            'cooling_system': (15.0, 10.0, -5.0),  # 5m altitude
            'waste_storage': (5.0, 15.0, -4.0),   # 4m altitude
        }
        
        target_location = task['location']
        if target_location in location_positions:
            x, y, z = location_positions[target_location]
            trajectory_msg.position = [x, y, z]
            trajectory_msg.velocity = [0.0, 0.0, 0.0]
            trajectory_msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
            self.trajectory_setpoint_pub.publish(trajectory_msg)
            
            self.mission_state = MissionState.MAPPING
    
    def switch_robot_type(self, new_type: RobotType):
        """Switch between different robot configurations"""
        self.get_logger().info(f"Switching from {self.current_robot_type} to {new_type}")
        
        if new_type == RobotType.AERIAL_DRONE:
            # Arm drone and switch to offboard mode
            self.arm_drone()
            
        elif new_type == RobotType.MANIPULATOR_ARM:
            # Initialize manipulator arm
            self.initialize_manipulator()
        
        self.current_robot_type = new_type
    
    def arm_drone(self):
        """Arm PX4 drone for flight operations"""
        arm_cmd = VehicleCommand()
        arm_cmd.command = VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM
        arm_cmd.param1 = 1.0  # Arm
        arm_cmd.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.vehicle_command_pub.publish(arm_cmd)
    
    def initialize_manipulator(self):
        """Initialize MoveIt manipulator arm"""
        # This would set up the manipulator for inspection tasks
        pass
    
    def robot_state_callback(self, msg):
        """Process robot state updates"""
        self.latest_robot_state = msg
    
    def radiation_field_callback(self, msg):
        """Process radiation field measurements"""
        # Update radiation data for planning
        self.latest_radiation_data = {
            'intensity': msg.intensity,
            'position': (msg.pose.position.x, msg.pose.position.y),
            'timestamp': msg.header.stamp
        }
    
    def safety_status_callback(self, msg):
        """Process safety status - highest priority"""
        if msg.emergency_stop_required:
            self.transition_to_emergency_stop()
        
        if msg.radiation_level > self.mission_config['max_radiation_exposure']:
            self.get_logger().warn(f"High radiation detected: {msg.radiation_level} mSv/h")
            self.safety_override = True
    
    def pose_callback(self, msg):
        """Process robot pose updates"""
        self.current_pose = msg
    
    def map_callback(self, msg):
        """Process map updates from SLAM"""
        self.current_map = msg
    
    def navigation_feedback_callback(self, feedback):
        """Process navigation feedback from Nav2"""
        # Handle TEB planner feedback
        pass
    
    def emergency_stop_callback(self, request, response):
        """Handle emergency stop service requests"""
        self.get_logger().critical("Emergency stop activated!")
        self.transition_to_emergency_stop()
        response.success = True
        response.message = "Emergency stop activated"
        return response
    
    def set_mission_callback(self, request, response):
        """Handle mission configuration requests"""
        try:
            mission_data = json.loads(request.mission_config)
            self.mission_config.update(mission_data)
            self.mission_state = MissionState.PLANNING
            response.success = True
            response.message = "Mission configured successfully"
        except Exception as e:
            response.success = False
            response.message = f"Mission configuration failed: {e}"
        
        return response
    
    def transition_to_emergency_stop(self):
        """Transition system to emergency stop state"""
        self.emergency_stop_active = True
        self.mission_state = MissionState.EMERGENCY_STOP
        
        # Stop all robot motion immediately
        stop_cmd = Twist()
        self.robot_commands_pub.publish(stop_cmd)
        
        # Alert safety systems
        alert_msg = String()
        alert_msg.data = "EMERGENCY STOP ACTIVATED - ALL SYSTEMS HALTED"
        self.safety_alert_pub.publish(alert_msg)
    
    def handle_emergency_stop(self):
        """Handle emergency stop state"""
        # Keep publishing stop commands
        stop_cmd = Twist()
        self.robot_commands_pub.publish(stop_cmd)
        
        # Check if emergency can be cleared
        if not self.safety_override:
            self.emergency_stop_active = False
            self.mission_state = MissionState.RETURNING_HOME
    
    def handle_idle_state(self):
        """Handle idle state - waiting for mission"""
        pass
    
    def handle_mapping_state(self):
        """Handle radiation mapping operations"""
        # Coordinate with radiation mapper
        self.mission_state = MissionState.INSPECTING
    
    def handle_inspection_state(self):
        """Handle detailed inspection operations"""
        # Execute inspection protocols
        self.plan_index += 1
        self.mission_state = MissionState.NAVIGATING
    
    def handle_return_home_state(self):
        """Handle return to home position"""
        # Navigate back to starting position
        self.mission_state = MissionState.MISSION_COMPLETE
    
    def publish_mission_status(self):
        """Publish current mission status"""
        status_msg = String()
        status_msg.data = f"State: {self.mission_state.value}, Robot: {self.current_robot_type.value}"
        self.mission_status_pub.publish(status_msg)

def main(args=None):
    rclpy.init(args=args)
    
    orchestrator = NuclearMissionOrchestrator()
    
    # Use MultiThreadedExecutor for concurrent operations
    from rclpy.executors import MultiThreadedExecutor
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(orchestrator)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        orchestrator.get_logger().info("Mission orchestrator shutting down")
    finally:
        orchestrator.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()