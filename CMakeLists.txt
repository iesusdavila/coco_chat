cmake_minimum_required(VERSION 3.8)
project(buddy_chat)

# ========== Configuración para Python ==========
find_package(rclpy REQUIRED)
find_package(sensor_msgs REQUIRED)

# Instalar el script Python
install(PROGRAMS
  scripts/buddy_chat.py
  scripts/stt_lifecycle_node.py
  scripts/llm_lifecycle_node.py
  scripts/tts_lifecycle_node.py
  scripts/control_manager_node.py
  DESTINATION lib/${PROJECT_NAME}
)

# ========== Instalación común ==========
install(DIRECTORY
  launch
  models
  scripts
  DESTINATION share/${PROJECT_NAME}
)

ament_export_dependencies(
  rclpy
  sensor_msgs
  std_msgs
)

ament_package()
