from launch import LaunchDescription
from launch_ros.actions import LifecycleNode
from launch_ros.actions import Node

def generate_launch_description():

    stt_node = LifecycleNode(
        name='stt_lifecycle_node',
        namespace='',
        package='buddy_chat',
        executable='stt_lifecycle_node.py',
        output='screen',
    )

    llm_node = LifecycleNode(
        name='llm_lifecycle_node',
        namespace='',
        package='buddy_chat',
        executable='llm_lifecycle_node',
        output='screen',
    )

    tts_node = LifecycleNode(
        name='tts_lifecycle_node',
        namespace='',
        package='buddy_chat',
        executable='tts_lifecycle_node.py',
        output='screen',
    )

    control_manager_node = Node(
        name='lifecycle_control_manager',
        package='buddy_chat',
        executable='control_manager_node.py',
        output='screen'
    )

    return LaunchDescription([
        stt_node,
        llm_node,
        tts_node,
        control_manager_node,
    ])