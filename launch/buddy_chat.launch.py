import os
from launch import LaunchDescription
from launch_ros.actions import LifecycleNode
from launch_ros.actions import Node
from launch.actions import RegisterEventHandler
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Get package share directory
    pkg_share_dir = get_package_share_directory('buddy_chat')

    # STT Lifecycle Node
    stt_node = LifecycleNode(
        name='stt_lifecycle_node',
        namespace='',
        package='buddy_chat',
        executable='stt_lifecycle_node.py',
        output='screen',
        parameters=[
            {'vosk_model_path': os.path.join(pkg_share_dir, 'models', 'STT', 'vosk-model-small-es-0.42')}
        ]
    )

    # LLM Lifecycle Node
    llm_node = LifecycleNode(
        name='llm_lifecycle_node',
        namespace='',
        package='buddy_chat',
        executable='llm_lifecycle_node.py',
        output='screen',
        parameters=[
            {'llm_model_path': os.path.join(pkg_share_dir, 'models', 'LLM/models--ggml-org--Meta-Llama-3.1-8B-Instruct-Q4_0-GGUF/snapshots/0aba27dd2f1c7f4941a94a5c59d80e0a256f9ff8', 'meta-llama-3.1-8b-instruct-q4_0.gguf')}
        ]
    )

    # TTS Lifecycle Node
    tts_node = LifecycleNode(
        name='tts_lifecycle_node',
        namespace='',
        package='buddy_chat',
        executable='tts_lifecycle_node.py',
        output='screen',
        parameters=[
            {'tts_model_path': os.path.join(pkg_share_dir, 'models', 'TTS', 'es_MX-claude-high.onnx')},
            {'tts_config_path': os.path.join(pkg_share_dir, 'models', 'TTS', 'es_MX-claude-high.onnx.json')}
        ]
    )

    # Control Manager Node
    control_manager_node = Node(
        name='lifecycle_control_manager',
        package='buddy_chat',
        executable='control_manager_node.py',
        output='screen'
    )

    # Create and return the launch description
    return LaunchDescription([
        # Nodes
        stt_node,
        llm_node,
        tts_node,
        control_manager_node,
    ])