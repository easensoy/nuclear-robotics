cmake_minimum_required(VERSION 3.8)
project(nuclear_robot_core)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(nav2_msgs REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(control_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(diagnostic_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(builtin_interfaces REQUIRED)

# Generate custom message and service files
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/RadiationField.msg"
  "msg/RobotState.msg"  
  "msg/SafetyStatus.msg"
  "srv/EmergencyStop.srv"
  "srv/SetMission.srv"
  DEPENDENCIES builtin_interfaces std_msgs geometry_msgs sensor_msgs
)

# Install Python modules
ament_python_install_package(${PROJECT_NAME})

# Install Python executables
install(PROGRAMS
  src/nuclear_robot_state_estimator.py
  src/nuclear_robot_controller.py
  src/nuclear_robot_system_identification.py
  src/nuclear_radiation_mapper.py
  src/nuclear_safety_monitor.py
  src/nuclear_mission_orchestrator.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
)

# Install config files
install(DIRECTORY
  config
  DESTINATION share/${PROJECT_NAME}/
)

# Install URDF and mesh files
install(DIRECTORY
  urdf
  meshes
  DESTINATION share/${PROJECT_NAME}/
)

# Install worlds and models for simulation
install(DIRECTORY
  worlds
  models
  DESTINATION share/${PROJECT_NAME}/
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
  
  # Python tests
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(nuclear_robot_tests tests/
    APPEND_ENV PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}
  )
endif()

# Export