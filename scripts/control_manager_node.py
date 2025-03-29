#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from lifecycle_msgs.msg import Transition
from lifecycle_msgs.srv import ChangeState
import threading

class LifecycleNodesManager(Node):
    def __init__(self):
        super().__init__('lifecycle_nodes_manager')
        
        # Topics to track STT and TTS status
        self.create_subscription(Bool, '/stt_terminado', self.stt_status_callback, 10)
        self.create_subscription(Bool, '/tts_terminado', self.tts_status_callback, 10)
        
        # Service clients for changing lifecycle node states
        self.stt_state_client = self.create_client(ChangeState, '/stt_lifecycle_node/change_state')
        self.llm_state_client = self.create_client(ChangeState, '/llm_lifecycle_node/change_state')
        self.tts_state_client = self.create_client(ChangeState, '/tts_lifecycle_node/change_state')
        
        # Node state tracking
        self.stt_terminated = False
        self.tts_terminated = True

        # Lock for thread-safe state management
        self.state_lock = threading.Lock()

        # Initialize lifecycle nodes
        self._configure_initial_nodes()

    def _configure_initial_nodes(self):
        """Configure STT and TTS nodes on startup"""
        threading.Thread(target=self._initial_configuration).start()

    def _initial_configuration(self):
        self.change_node_state('/stt_lifecycle_node', Transition.TRANSITION_CONFIGURE)
        self.change_node_state('/stt_lifecycle_node', Transition.TRANSITION_ACTIVATE)
        self.change_node_state('/llm_lifecycle_node', Transition.TRANSITION_CONFIGURE)
        self.change_node_state('/tts_lifecycle_node', Transition.TRANSITION_CONFIGURE)
    
    def stt_status_callback(self, msg):
        """Handle STT status changes"""
        with self.state_lock:
            self.stt_terminated = msg.data
            self.manage_node_lifecycle()
    
    def tts_status_callback(self, msg):
        """Handle TTS status changes"""
        with self.state_lock:
            self.tts_terminated = msg.data
            self.manage_node_lifecycle()
    
    def manage_node_lifecycle(self):
        """Manage nodes based on STT and TTS status"""
        threading.Thread(target=self._manage_lifecycle_thread).start()

    def _manage_lifecycle_thread(self):
        """Thread-safe lifecycle management"""
        with self.state_lock:
            self.get_logger().info(f"STT: {self.stt_terminated}, TTS: {self.tts_terminated}")
            
            if self.stt_terminated and not self.tts_terminated:
                # Change states in a separate thread
                self.change_node_state('/llm_lifecycle_node', Transition.TRANSITION_ACTIVATE)
                self.change_node_state('/tts_lifecycle_node', Transition.TRANSITION_ACTIVATE)
                self.change_node_state('/stt_lifecycle_node', Transition.TRANSITION_DEACTIVATE)
            
            elif self.tts_terminated and not self.stt_terminated:
                # Activate STT, Deactivate LLM
                # self.change_node_state('/stt_lifecycle_node', Transition.TRANSITION_CONFIGURE)
                self.change_node_state('/stt_lifecycle_node', Transition.TRANSITION_ACTIVATE)
                self.change_node_state('/tts_lifecycle_node', Transition.TRANSITION_DEACTIVATE)
                self.change_node_state('/llm_lifecycle_node', Transition.TRANSITION_DEACTIVATE)
    
    def change_node_state(self, node_name, transition_id):
        """Asynchronous node state change"""
        self.get_logger().info(f"Changing state of {node_name} to {transition_id}")

        req = ChangeState.Request()
        req.transition.id = transition_id
        
        # Select correct client based on node name
        if "stt" in node_name:
            client = self.stt_state_client
        elif "llm" in node_name:
            client = self.llm_state_client
        elif "tts" in node_name:
            client = self.tts_state_client
        
        def call_service():
            if not client.wait_for_service(timeout_sec=5.0):
                self.get_logger().warn(f"Service for {node_name} not available")
                return 
            future = client.call_async(req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        
        threading.Thread(target=call_service).start()

def main(args=None):
    rclpy.init(args=args)
    manager = LifecycleNodesManager()
    rclpy.spin(manager)
    rclpy.shutdown()

if __name__ == '__main__':
    main()