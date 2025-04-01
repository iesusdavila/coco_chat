cmake_minimum_required(VERSION 3.8)
project(buddy_chat)

# ========== Configuración para C++ ==========
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(buddy_interfaces REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(rclcpp_action REQUIRED)

# Incluir directorios de librerías
include_directories(
  include
  /usr/local/include 
)

# Configuración del ejecutable C++
add_executable(llm_lifecycle_node src/llm_lifecycle_node.cpp)
ament_target_dependencies(llm_lifecycle_node 
  rclcpp 
  sensor_msgs 
  buddy_interfaces 
  std_msgs
  rclcpp_lifecycle
  rclcpp_action
)

# Añadir el directorio de la librería llamcpp
link_directories(/usr/local/lib)

# Enlazar las bibliotecas necesarias
target_link_libraries(llm_lifecycle_node
  /usr/local/lib/libllama.so 
  /usr/local/lib/libggml.so
)

# ========== Configuración para Python ==========
find_package(rclpy REQUIRED)
find_package(sensor_msgs REQUIRED)

# Instalar el script Python
install(PROGRAMS
  scripts/buddy_chat.py
  scripts/stt_lifecycle_node.py
  # scripts/llm_lifecycle_node.py
  scripts/tts_lifecycle_node.py
  scripts/control_manager_node.py
  DESTINATION lib/${PROJECT_NAME}
)

# ========== Instalación común ==========
install(DIRECTORY
  launch
  models
  scripts
  src
  DESTINATION share/${PROJECT_NAME}
)

install(TARGETS 
  llm_lifecycle_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_export_dependencies(
  rclpy
  sensor_msgs
  std_msgs
  buddy_interfaces
  rclcpp_lifecycle
  rclcpp_action
)

ament_export_include_directories(include)
ament_export_libraries(llm_lifecycle_node)
ament_package()
