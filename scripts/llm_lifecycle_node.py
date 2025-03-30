#!/usr/bin/env python3

import rclpy
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn 
from rclpy.action import ActionServer
from buddy_interfaces.msg import PersonResponse, LLMStatus
from buddy_interfaces.action import ProcessResponse
from llama_cpp import Llama
from ament_index_python.packages import get_package_share_directory
import os
import re

class TextProcessor:
    """Clase para procesar y limpiar texto para TTS"""
    @staticmethod
    def clean_text(text):
        """Limpia el texto y prepara para síntesis de voz"""
        text = re.sub(r'[!¡?¿*,.:;()\[\]{}]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return re.split(r'(?<=[.!?])\s+', text)
    
class LLMLifecycleNode(LifecycleNode):
    def __init__(self):
        super().__init__('llm_lifecycle_node')
        
        # Configure model paths
        pkg_share_dir = get_package_share_directory('buddy_chat')
        self.llm_model_path = os.path.join(pkg_share_dir, 'models', 'LLM/models--ggml-org--Meta-Llama-3.1-8B-Instruct-Q4_0-GGUF/snapshots/0aba27dd2f1c7f4941a94a5c59d80e0a256f9ff8', 'meta-llama-3.1-8b-instruct-q4_0.gguf')
        
        # Subscribers
        self.create_subscription(PersonResponse, '/response_person', self.process_input, 10)
        
        # Publishers
        self.llm_status_publisher = self.create_publisher(LLMStatus, '/llm_status', 10)
        
        # Action Server
        self._action_server = ActionServer(
            self,
            ProcessResponse,
            '/response_llama',
            self.execute_response_generation
        )
        
        # LLM Model
        self.model = None
        self.conversation_history = []
        
    def on_configure(self, state):
        self.get_logger().info('Configuring LLM Node')
        
        try:
            # Load LLAMA3 model
            self.model = Llama(
                model_path=self.llm_model_path,
                n_ctx=2048,
                verbose=False,
                use_gpu=True,
                n_gpu_layers=10,
                threads=4
            )
            
            return TransitionCallbackReturn.SUCCESS
        except Exception as e:
            self.get_logger().error(f'Failed to configure: {str(e)}')
            return TransitionCallbackReturn.FAILURE
    
    def on_activate(self, state):
        self.get_logger().info('Activating LLM Node')
        return TransitionCallbackReturn.SUCCESS
    
    def on_deactivate(self, state):
        self.get_logger().info('Deactivating LLM Node')
        return TransitionCallbackReturn.SUCCESS
    
    def process_input(self, msg):
        """Process input from person response topic"""
        self.get_logger().info(f"Procesando mensaje de la persona: {msg.text}")

        status_msg = LLMStatus()
        status_msg.is_processing = True
        status_msg.current_response = msg.text
        status_msg.timestamp = self.get_clock().now().to_msg()
        self.llm_status_publisher.publish(status_msg)
    
    def execute_response_generation(self, goal_handle):
        """Generate response using LLAMA3"""
        feedback_msg = ProcessResponse.Feedback()
        result_msg = ProcessResponse.Result()
        
        system_prompt = """
        Eres Leo, un asistente virtual amigable para niños que se encuentran en hospitales 
        y tienen entre 7 a 12 años.
        """
        
        self.conversation_history = [{"role": "system", "content": system_prompt}]
        self.conversation_history.append({"role": "user", "content": goal_handle.request.input_text})
        
        response_stream = self.model.create_chat_completion(
            messages=self.conversation_history,
            max_tokens=300,
            temperature=0.7,
            stream=True
        )
        
        buffer = ""
        full_response = ""
        self.get_logger().info("Generando respuesta...")
        self.get_logger().info(f"Mensaje de entrada: {goal_handle.request.input_text}")
        
        for token in response_stream:
            content = token['choices'][0]['delta'].get('content', '')
            buffer += content
            full_response += content

            if len(buffer) > 0 and content.endswith((' ', '.', ',', '!', '?')):
                for phrase in TextProcessor.clean_text(buffer):
                    if phrase:
                        feedback_msg.current_chunk = phrase
                        self.get_logger().info(f"Chunk procesado: {phrase}")
                        feedback_msg.progress = 0.0  # Normalized progress
                        goal_handle.publish_feedback(feedback_msg)
                        
                buffer = ""
            
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                result_msg.full_response = "Proceso cancelado"
                result_msg.completed = False
                return result_msg
        
        self.get_logger().info("Toda la respuesta generada por el modelo")
        feedback_msg.current_chunk = "[END_FINAL]"
        feedback_msg.progress = 1.0
        goal_handle.publish_feedback(feedback_msg)

        
        # Add response to conversation history
        self.conversation_history.append({"role": "assistant", "content": full_response})

        goal_handle.succeed()
        
        result_msg.full_response = full_response
        result_msg.completed = True
        
        return result_msg

def main(args=None):
    rclpy.init(args=args)
    node = LLMLifecycleNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()