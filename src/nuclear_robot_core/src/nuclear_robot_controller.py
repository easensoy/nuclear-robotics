#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
import numpy as np
import threading
import time
from enum import Enum
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json

# ROS2 message imports
from std_msgs.msg import Bool, String, Float64, Float64MultiArray
from geometry_msgs.msg import PoseStamped, Twist
from sensor_msgs.msg import BatteryState, Temperature
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from nav_msgs.msg import Odometry

# Custom message imports
from nuclear_robot_core.msg import SafetyStatus, RobotState, RadiationField
from nuclear_robot_core.srv import EmergencyStop

class SafetyLevel(Enum):
    NORMAL = "normal"
    CAUTION = "caution"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class SafetyViolationType(Enum):
    RADIATION_EXPOSURE = "radiation_exposure"
    BATTERY_CRITICAL = "battery_critical"
    SYSTEM_TEMPERATURE = "system_temperature"
    COMMUNICATION_LOSS = "communication_loss"
    COLLISION_IMMINENT = "collision_imminent"
    MISSION_TIMEOUT = "mission_timeout"
    HARDWARE_FAILURE = "hardware_failure"
    POSITION_UNCERTAINTY = "position_uncertainty"

@dataclass
class SafetyThresholds:
    """Safety threshold configuration for nuclear operations"""
    max_radiation_exposure: float = 10.0    # mSv/h - regulatory limit
    max_cumulative_dose: float = 100.0      # mSv - mission total
    min_battery_level: float = 20.0         # percentage
    max_system_temperature: float = 85.0    # Celsius
    max_communication_gap: float = 5.0      # seconds
    min_obstacle_distance: float = 0.3      # meters
    max_mission_duration: float = 14400.0   # seconds (4 hours)
    max_position_uncertainty: float = 2.0   # meters
    max_velocity_emergency: float = 0.1     # m/s during emergency
    
class SafetyViolation:
    """Individual safety violation record"""
    
    def __init__(self, violation_type: SafetyViolationType, 
                 severity: SafetyLevel, description: str, 
                 timestamp: float, value: float = None):
        self.violation_type = violation_type
        self.severity = severity
        self.description = description
        self.timestamp = timestamp
        self.value = value
        self.acknowledged = False
        self.resolved = False

class NuclearSafetyMonitor(Node):
    """Comprehensive safety monitoring system for nuclear robotics operations"""
    
    def __init__(self):
        super().__init__('nuclear_safety_monitor')
        
        # Safety configuration
        self.safety_thresholds = SafetyThresholds()
        self.load_safety_configuration()
        
        # Safety state tracking
        self.current_safety_level = SafetyLevel.NORMAL
        self.active_violations = {}
        self.violation_history = []
        self.emergency_stop_active = False
        self.safety_override_authorized = False
        
        # Mission tracking
        self.mission_start_time = None
        self.cumulative_radiation_dose = 0.0
        self.last_heartbeat = time.time()
        
        # System state monitoring
        self.robot_state = None
        self.current_pose = None
        self.current_velocity = None
        self.battery_status = None
        self.system_temperatures = {}
        self.radiation_readings = []
        
        # Callback groups for different priority levels
        self.critical_group = ReentrantCallbackGroup()
        self.monitoring_group = ReentrantCallbackGroup()
        self.diagnostic_group = ReentrantCallbackGroup()
        
        self.setup_publishers()
        self.setup_subscribers()
        self.setup_services()
        
        # Safety monitoring timers with different frequencies
        self.critical_monitor_timer = self.create_timer(
            0.1, self.critical_safety_check, callback_group=self.critical_group)
        self.general_monitor_timer = self.create_timer(
            1.0, self.general_safety_check, callback_group=self.monitoring_group)
        self.diagnostic_timer = self.create_timer(
            5.0, self.publish_diagnostics, callback_group=self.diagnostic_group)
        
        # Watchdog timer for communication monitoring
        self.watchdog_timer = self.create_timer(0.5, self.communication_watchdog)
        
        self.get_logger().info("Nuclear Safety Monitor initialized with regulatory compliance")
    
    def load_safety_configuration(self):
        """Load safety thresholds from configuration"""
        try:
            # In production, this would load from a YAML configuration file
            # Here we use default values with nuclear industry standards
            self.safety_thresholds.max_radiation_exposure = 10.0  # ALARA principle
            self.safety_thresholds.max_cumulative_dose = 50.0     # Conservative limit
            self.get_logger().info("Safety configuration loaded with nuclear industry standards")
        except Exception as e:
            self.get_logger().error(f"Failed to load safety configuration: {e}")
    
    def setup_publishers(self):
        """Initialize safety monitoring publishers"""
        # High-priority safety status
        self.safety_status_pub = self.create_publisher(
            SafetyStatus, '/safety/status', 10)
        
        # Emergency stop command
        self.emergency_stop_pub = self.create_publisher(
            Bool, '/emergency_stop', 10)
        
        # Safety alerts for operators
        self.safety_alerts_pub = self.create_publisher(
            String, '/safety/alerts', 10)
        
        # Diagnostic information
        self.diagnostics_pub = self.create_publisher(
            DiagnosticArray, '/diagnostics', 10)
        
        # Safety-controlled robot commands
        self.safe_cmd_vel_pub = self.create_publisher(
            Twist, '/robot/cmd_vel_safe', 10)
        
        # Mission abort command
        self.mission_abort_pub = self.create_publisher(
            Bool, '/mission/abort', 10)
    
    def setup_subscribers(self):
        """Initialize safety monitoring subscribers"""
        # Critical system inputs with high priority
        self.robot_state_sub = self.create_subscription(
            RobotState, '/robot/state', self.robot_state_callback, 10,
            callback_group=self.critical_group)
        
        self.radiation_sub = self.create_subscription(
            RadiationField, '/radiation/field', self.radiation_callback, 10,
            callback_group=self.critical_group)
        
        # System monitoring inputs
        self.pose_sub = self.create_subscription(
            PoseStamped, '/robot/pose_estimated', self.pose_callback, 10)
        
        self.odom_sub = self.create_subscription(
            Odometry, '/robot/odometry_filtered', self.odometry_callback, 10)
        
        self.battery_sub = self.create_subscription(
            BatteryState, '/robot/battery', self.battery_callback, 10)
        
        self.temperature_sub = self.create_subscription(
            Temperature, '/robot/temperature', self.temperature_callback, 10)
        
        # Command monitoring for safety override
        self.cmd_vel_sub = self.create_subscription(
            Twist, '/robot/cmd_vel', self.cmd_vel_callback, 10,
            callback_group=self.critical_group)
        
        # Heartbeat from main system
        self.heartbeat_sub = self.create_subscription(
            String, '/system/heartbeat', self.heartbeat_callback, 10)
    
    def setup_services(self):
        """Initialize safety services"""
        # Emergency stop service
        self.emergency_stop_service = self.create_service(
            EmergencyStop, '/safety/emergency_stop', 
            self.emergency_stop_service_callback,
            callback_group=self.critical_group)
        
        # Safety override authorization service
        self.safety_override_service = self.create_service(
            EmergencyStop, '/safety/authorize_override',
            self.safety_override_callback)
    
    def robot_state_callback(self, msg):
        """Monitor robot state for safety violations"""
        self.robot_state = msg
        self.last_heartbeat = time.time()
        
        # Check mission duration
        if msg.mission_status == "active" and self.mission_start_time is None:
            self.mission_start_time = time.time()
        
        # Monitor for system failures
        if msg.system_health == "degraded":
            self.add_safety_violation(
                SafetyViolationType.HARDWARE_FAILURE,
                SafetyLevel.WARNING,
                f"System health degraded: {msg.error_message}"
            )
        elif msg.system_health == "failed":
            self.add_safety_violation(
                SafetyViolationType.HARDWARE_FAILURE,
                SafetyLevel.CRITICAL,
                f"System health failed: {msg.error_message}"
            )
    
    def radiation_callback(self, msg):
        """Monitor radiation exposure levels"""
        self.radiation_readings.append({
            'intensity': msg.intensity,
            'timestamp': time.time(),
            'position': (msg.pose.position.x, msg.pose.position.y, msg.pose.position.z)
        })
        
        # Keep only recent readings (last 10 minutes)
        current_time = time.time()
        self.radiation_readings = [
            r for r in self.radiation_readings 
            if current_time - r['timestamp'] < 600
        ]
        
        # Update cumulative dose (simplified calculation)
        if len(self.radiation_readings) > 1:
            time_delta = self.radiation_readings[-1]['timestamp'] - self.radiation_readings[-2]['timestamp']
            dose_increment = msg.intensity * (time_delta / 3600.0)  # mSv
            self.cumulative_radiation_dose += dose_increment
        
        # Check immediate radiation exposure
        if msg.intensity > self.safety_thresholds.max_radiation_exposure:
            severity = SafetyLevel.CRITICAL if msg.intensity > 20.0 else SafetyLevel.WARNING
            self.add_safety_violation(
                SafetyViolationType.RADIATION_EXPOSURE,
                severity,
                f"Radiation exposure: {msg.intensity:.2f} mSv/h exceeds limit",
                value=msg.intensity
            )
        
        # Check cumulative dose
        if self.cumulative_radiation_dose > self.safety_thresholds.max_cumulative_dose:
            self.add_safety_violation(
                SafetyViolationType.RADIATION_EXPOSURE,
                SafetyLevel.CRITICAL,
                f"Cumulative dose: {self.cumulative_radiation_dose:.2f} mSv exceeds mission limit",
                value=self.cumulative_radiation_dose
            )
    
    def pose_callback(self, msg):
        """Monitor position and localization uncertainty"""
        self.current_pose = msg
        
        # Check position uncertainty from covariance
        position_cov = msg.pose.covariance
        uncertainty = np.sqrt(position_cov[0] + position_cov[7])  # x,y uncertainty
        
        if uncertainty > self.safety_thresholds.max_position_uncertainty:
            self.add_safety_violation(
                SafetyViolationType.POSITION_UNCERTAINTY,
                SafetyLevel.WARNING,
                f"Position uncertainty: {uncertainty:.2f}m exceeds limit",
                value=uncertainty
            )
    
    def odometry_callback(self, msg):
        """Monitor robot velocity and motion"""
        self.current_velocity = msg.twist.twist
        
        # Calculate current speed
        linear_speed = np.sqrt(
            msg.twist.twist.linear.x**2 + 
            msg.twist.twist.linear.y**2
        )
        
        # Emergency speed limit check
        if (self.emergency_stop_active and 
            linear_speed > self.safety_thresholds.max_velocity_emergency):
            self.add_safety_violation(
                SafetyViolationType.COLLISION_IMMINENT,
                SafetyLevel.EMERGENCY,
                f"Speed {linear_speed:.2f}m/s exceeds emergency limit",
                value=linear_speed
            )
    
    def battery_callback(self, msg):
        """Monitor battery status for mission safety"""
        self.battery_status = msg
        
        if msg.percentage < self.safety_thresholds.min_battery_level:
            severity = SafetyLevel.CRITICAL if msg.percentage < 10.0 else SafetyLevel.WARNING
            self.add_safety_violation(
                SafetyViolationType.BATTERY_CRITICAL,
                severity,
                f"Battery level: {msg.percentage:.1f}% below safety threshold",
                value=msg.percentage
            )
    
    def temperature_callback(self, msg):
        """Monitor system temperature"""
        self.system_temperatures['main'] = msg.temperature
        
        if msg.temperature > self.safety_thresholds.max_system_temperature:
            self.add_safety_violation(
                SafetyViolationType.SYSTEM_TEMPERATURE,
                SafetyLevel.WARNING,
                f"System temperature: {msg.temperature:.1f}°C exceeds limit",
                value=msg.temperature
            )
    
    def cmd_vel_callback(self, msg):
        """Monitor and potentially override velocity commands for safety"""
        if not self.emergency_stop_active:
            # Pass through commands during normal operation
            self.safe_cmd_vel_pub.publish(msg)
        else:
            # Override with stop command during emergency
            stop_cmd = Twist()
            self.safe_cmd_vel_pub.publish(stop_cmd)
    
    def heartbeat_callback(self, msg):
        """Update system heartbeat timestamp"""
        self.last_heartbeat = time.time()
    
    def critical_safety_check(self):
        """High-frequency critical safety monitoring (10Hz)"""
        current_time = time.time()
        
        # Check for emergency stop conditions
        critical_violations = [
            v for v in self.active_violations.values() 
            if v.severity in [SafetyLevel.CRITICAL, SafetyLevel.EMERGENCY]
        ]
        
        if critical_violations and not self.emergency_stop_active:
            self.trigger_emergency_stop("Critical safety violations detected")
        
        # Publish current safety status
        self.publish_safety_status()
    
    def general_safety_check(self):
        """General safety monitoring (1Hz)"""
        current_time = time.time()
        
        # Check communication timeout
        if current_time - self.last_heartbeat > self.safety_thresholds.max_communication_gap:
            self.add_safety_violation(
                SafetyViolationType.COMMUNICATION_LOSS,
                SafetyLevel.CRITICAL,
                f"Communication loss: {current_time - self.last_heartbeat:.1f}s",
                value=current_time - self.last_heartbeat
            )
        
        # Check mission timeout
        if (self.mission_start_time and 
            current_time - self.mission_start_time > self.safety_thresholds.max_mission_duration):
            self.add_safety_violation(
                SafetyViolationType.MISSION_TIMEOUT,
                SafetyLevel.WARNING,
                f"Mission duration: {(current_time - self.mission_start_time)/3600:.1f}h exceeds limit"
            )
        
        # Update overall safety level
        self.update_safety_level()
        
        # Clean up resolved violations
        self.cleanup_resolved_violations()
    
    def communication_watchdog(self):
        """Monitor system communication health"""
        current_time = time.time()
        
        # Check if we haven't received heartbeat recently
        if current_time - self.last_heartbeat > 10.0:  # 10 second timeout
            if not self.emergency_stop_active:
                self.get_logger().error("Communication watchdog triggered emergency stop")
                self.trigger_emergency_stop("Communication timeout - watchdog activated")
    
    def add_safety_violation(self, violation_type: SafetyViolationType, 
                           severity: SafetyLevel, description: str, value: float = None):
        """Add or update safety violation"""
        violation_key = violation_type.value
        current_time = time.time()
        
        # Create new violation or update existing
        violation = SafetyViolation(violation_type, severity, description, current_time, value)
        self.active_violations[violation_key] = violation
        self.violation_history.append(violation)
        
        # Limit violation history size
        if len(self.violation_history) > 1000:
            self.violation_history = self.violation_history[-800:]
        
        # Log violation
        log_func = self.get_logger().error if severity in [SafetyLevel.CRITICAL, SafetyLevel.EMERGENCY] else self.get_logger().warn
        log_func(f"Safety violation - {violation_type.value}: {description}")
        
        # Publish safety alert
        alert_msg = String()
        alert_msg.data = f"[{severity.value.upper()}] {violation_type.value}: {description}"
        self.safety_alerts_pub.publish(alert_msg)
    
    def update_safety_level(self):
        """Update overall system safety level"""
        if not self.active_violations:
            self.current_safety_level = SafetyLevel.NORMAL
        else:
            # Set to highest severity level among active violations
            severity_levels = [v.severity for v in self.active_violations.values()]
            
            if SafetyLevel.EMERGENCY in severity_levels:
                self.current_safety_level = SafetyLevel.EMERGENCY
            elif SafetyLevel.CRITICAL in severity_levels:
                self.current_safety_level = SafetyLevel.CRITICAL
            elif SafetyLevel.WARNING in severity_levels:
                self.current_safety_level = SafetyLevel.WARNING
            else:
                self.current_safety_level = SafetyLevel.CAUTION
    
    def cleanup_resolved_violations(self):
        """Remove resolved safety violations"""
        current_time = time.time()
        resolved_keys = []
        
        for key, violation in self.active_violations.items():
            # Check if violation conditions are resolved
            if self.is_violation_resolved(violation):
                violation.resolved = True
                resolved_keys.append(key)
        
        # Remove resolved violations
        for key in resolved_keys:
            del self.active_violations[key]
            self.get_logger().info(f"Safety violation resolved: {key}")
    
    def is_violation_resolved(self, violation: SafetyViolation) -> bool:
        """Check if a safety violation has been resolved"""
        current_time = time.time()
        
        # Auto-resolve old violations (may have been temporary)
        if current_time - violation.timestamp > 30.0:  # 30 seconds
            return True
        
        # Check specific resolution conditions
        if violation.violation_type == SafetyViolationType.RADIATION_EXPOSURE:
            recent_readings = [r for r in self.radiation_readings if current_time - r['timestamp'] < 5.0]
            if recent_readings:
                avg_radiation = np.mean([r['intensity'] for r in recent_readings])
                return avg_radiation <= self.safety_thresholds.max_radiation_exposure
        
        elif violation.violation_type == SafetyViolationType.BATTERY_CRITICAL:
            if self.battery_status:
                return self.battery_status.percentage >= self.safety_thresholds.min_battery_level
        
        elif violation.violation_type == SafetyViolationType.SYSTEM_TEMPERATURE:
            if 'main' in self.system_temperatures:
                return self.system_temperatures['main'] <= self.safety_thresholds.max_system_temperature
        
        elif violation.violation_type == SafetyViolationType.COMMUNICATION_LOSS:
            return current_time - self.last_heartbeat <= self.safety_thresholds.max_communication_gap
        
        return False
    
    def trigger_emergency_stop(self, reason: str):
        """Trigger emergency stop procedure"""
        if self.emergency_stop_active:
            return  # Already in emergency stop
        
        self.emergency_stop_active = True
        self.get_logger().critical(f"EMERGENCY STOP TRIGGERED: {reason}")
        
        # Publish emergency stop command
        emergency_msg = Bool()
        emergency_msg.data = True
        self.emergency_stop_pub.publish(emergency_msg)
        
        # Abort current mission
        abort_msg = Bool()
        abort_msg.data = True
        self.mission_abort_pub.publish(abort_msg)
        
        # Stop robot motion immediately
        stop_cmd = Twist()
        self.safe_cmd_vel_pub.publish(stop_cmd)
        
        # Add emergency violation
        self.add_safety_violation(
            SafetyViolationType.HARDWARE_FAILURE,
            SafetyLevel.EMERGENCY,
            f"Emergency stop activated: {reason}"
        )
    
    def publish_safety_status(self):
        """Publish current safety status"""
        msg = SafetyStatus()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        
        msg.safety_level = self.current_safety_level.value
        msg.emergency_stop_required = self.emergency_stop_active
        msg.radiation_level = self.radiation_readings[-1]['intensity'] if self.radiation_readings else 0.0
        msg.cumulative_dose = self.cumulative_radiation_dose
        msg.battery_level = self.battery_status.percentage if self.battery_status else 0.0
        msg.system_temperature = self.system_temperatures.get('main', 0.0)
        msg.active_violations = len(self.active_violations)
        
        # Mission timing
        if self.mission_start_time:
            msg.mission_elapsed_time = time.time() - self.mission_start_time
        else:
            msg.mission_elapsed_time = 0.0
        
        # Communication status
        msg.last_heartbeat = time.time() - self.last_heartbeat
        
        self.safety_status_pub.publish(msg)
    
    def publish_diagnostics(self):
        """Publish detailed diagnostic information"""
        diag_array = DiagnosticArray()
        diag_array.header.stamp = self.get_clock().now().to_msg()
        
        # Overall safety status
        safety_status = DiagnosticStatus()
        safety_status.name = "Nuclear Safety Monitor"
        safety_status.hardware_id = "nuclear_robot_safety"
        
        if self.current_safety_level == SafetyLevel.NORMAL:
            safety_status.level = DiagnosticStatus.OK
            safety_status.message = "All safety systems normal"
        elif self.current_safety_level in [SafetyLevel.CAUTION, SafetyLevel.WARNING]:
            safety_status.level = DiagnosticStatus.WARN
            safety_status.message = f"Safety level: {self.current_safety_level.value}"
        else:
            safety_status.level = DiagnosticStatus.ERROR
            safety_status.message = f"CRITICAL: Safety level {self.current_safety_level.value}"
        
        # Add diagnostic values
        safety_status.values.extend([
            KeyValue(key="safety_level", value=self.current_safety_level.value),
            KeyValue(key="active_violations", value=str(len(self.active_violations))),
            KeyValue(key="cumulative_dose_mSv", value=f"{self.cumulative_radiation_dose:.2f}"),
            KeyValue(key="emergency_stop_active", value=str(self.emergency_stop_active)),
            KeyValue(key="mission_elapsed_hours", 
                    value=f"{(time.time() - self.mission_start_time)/3600:.1f}" if self.mission_start_time else "0.0")
        ])
        
        diag_array.status.append(safety_status)
        
        # Individual system diagnostics
        for violation_type, violation in self.active_violations.items():
            violation_status = DiagnosticStatus()
            violation_status.name = f"Safety Violation: {violation_type}"
            violation_status.hardware_id = "nuclear_robot_safety"
            violation_status.level = DiagnosticStatus.ERROR if violation.severity in [SafetyLevel.CRITICAL, SafetyLevel.EMERGENCY] else DiagnosticStatus.WARN
            violation_status.message = violation.description
            
            violation_status.values.append(
                KeyValue(key="severity", value=violation.severity.value)
            )
            if violation.value is not None:
                violation_status.values.append(
                    KeyValue(key="value", value=str(violation.value))
                )
            
            diag_array.status.append(violation_status)
        
        self.diagnostics_pub.publish(diag_array)
    
    def emergency_stop_service_callback(self, request, response):
        """Handle emergency stop service requests"""
        if request.stop_requested:
            self.trigger_emergency_stop(f"Service request: {request.reason}")
            response.success = True
            response.message = "Emergency stop activated"
        else:
            # Attempt to clear emergency stop
            if self.safety_override_authorized:
                self.emergency_stop_active = False
                self.active_violations.clear()
                response.success = True
                response.message = "Emergency stop cleared with authorization"
            else:
                response.success = False
                response.message = "Emergency stop clear requires safety override authorization"
        
        return response
    
    def safety_override_callback(self, request, response):
        """Handle safety override authorization requests"""
        # In production, this would require proper authentication and logging
        self.safety_override_authorized = True
        self.get_logger().warn("Safety override authorized - use with extreme caution")
        
        response.success = True
        response.message = "Safety override authorized"
        return response

def main(args=None):
    rclpy.init(args=args)
    
    safety_monitor = NuclearSafetyMonitor()
    
    # Use MultiThreadedExecutor for concurrent safety monitoring
    from rclpy.executors import MultiThreadedExecutor
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(safety_monitor)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        safety_monitor.get_logger().info("Nuclear Safety Monitor shutting down")
    finally:
        safety_monitor.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()