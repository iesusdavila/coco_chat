#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.lifecycle import LifecycleNode
from rclpy.lifecycle import TransitionCallbackReturn 
from rclpy.lifecycle.node import LifecycleState
from lifecycle_msgs.msg import Transition
from std_msgs.msg import Bool
from buddy_interfaces.action import ProcessResponse
from ament_index_python.packages import get_package_share_directory
from piper import PiperVoice
import os
import threading
import time
import tempfile
import wave
from playsound import playsound
from buddy_interfaces.msg import LLMStatus
import queue

class TTSLifecycleNode(LifecycleNode):
    def __init__(self):
        super().__init__('tts_lifecycle_node')
        
        # Configure model paths
        pkg_share_dir = get_package_share_directory('buddy_chat')
        self.tts_model_path = os.path.join(pkg_share_dir, 'models', 'TTS', 'es_MX-claude-high.onnx')
        self.tts_config_path = os.path.join(pkg_share_dir, 'models', 'TTS', 'es_MX-claude-high.onnx.json')
        
        # TTS Voice
        self.voice = None
        
        # TTS Status Publisher
        self.tts_status_publisher = self.create_publisher(Bool, '/tts_terminado', 10)
        self.stt_status_publisher = self.create_publisher(Bool, '/stt_terminado', 10)

        # Subscribers
        self.create_subscription(LLMStatus, '/llm_status', self.process_input_person, 10)
        self.text_person = None
        
        # Action Client for LLAMA response
        self._action_client = None
    
    def process_input_person(self, msg):
        """Process input from person response topic"""
        self.get_logger().info('TTS is ready to speak')
        self.get_logger().info(f'Processing input: {msg.current_response}')
        self.text_person = msg.current_response
    
    def on_configure(self, state):
        self.get_logger().info('Configuring TTS Node')
        
        try:
            # Load Piper Voice model
            self.voice = PiperVoice.load(
                model_path=self.tts_model_path,
                config_path=self.tts_config_path,
                use_cuda=True
            )
            
            return TransitionCallbackReturn.SUCCESS
        except Exception as e:
            self.get_logger().error(f'Failed to configure: {str(e)}')
            return TransitionCallbackReturn.FAILURE
    
    def on_activate(self, state):
        self.get_logger().info('Activating TTS Node')
        
        # Setup action client to listen for LLAMA responses
        self._action_client = ActionClient(self, ProcessResponse, '/response_llama')
        
        self.audio_queue = queue.Queue()

        # Start listening for LLAMA responses
        goal_thread = threading.Thread(target=self._listen_and_speak)
        goal_thread.start()
        
        return TransitionCallbackReturn.SUCCESS
    
    def on_deactivate(self, state):
        self.get_logger().info('Deactivating TTS Node')
        return TransitionCallbackReturn.SUCCESS
    
    def _listen_and_speak(self):
        """Listen for LLAMA responses and convert to speech"""
        while rclpy.ok():
            if not self._action_client.wait_for_server(timeout_sec=1.0):
                self.get_logger().warn('Action server not available, retrying...')
                continue
            
            # Send goal to get response
            goal_msg = ProcessResponse.Goal()
            if self.text_person is not None:
                goal_msg.input_text = self.text_person
            else:
                return 
            # goal_msg.input_text = "Generar respuesta"

            self._action_client.wait_for_server()
            
            self.get_logger().info('Sending goal to LLAMA response')
            # Send goal and wait for result
            future = self._action_client.send_goal_async(
                goal_msg,
                feedback_callback=self._feedback_callback
            )

            self.get_logger().info('Waiting for goal to complete')
            
            # Wait for goal to complete
            # future.add_done_callback(self.result_callback)
            rclpy.spin_until_future_complete(self, future)
    
    def _feedback_callback(self, feedback_msg):
        """Process feedback and convert text to speech"""
        chunk = feedback_msg.feedback.current_chunk
        # self.audio_queue.put(chunk)
        
        if chunk and chunk != "[END_FINAL]":
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as fp:
                with wave.open(fp.name, 'wb') as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(self.voice.config.sample_rate)
                    self.voice.synthesize(chunk, wav_file)
                
                # Play audio
                playsound(fp.name)
        
        # If complete and playsound is not playing, publish TTS status
        if feedback_msg.feedback.progress == 1.0:
            self.get_logger().info('TTS synthesis complete')
            status_msg = Bool()
            status_msg.data = True
            self.tts_status_publisher.publish(status_msg)

            # Publish STT status
            stt_status_msg = Bool()
            stt_status_msg.data = False
            self.stt_status_publisher.publish(stt_status_msg)
    
    def result_callback(self, future):
        """Procesa el resultado final de la acción"""
        result = future.result()
        if result:
            self.get_logger().info('Objetivo completado con éxito')
        else:
            self.get_logger().error('Error en la acción de LLAMA')

def main(args=None):
    rclpy.init(args=args)
    node = TTSLifecycleNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()