#!/usr/bin/env python3

import re
import os
import rclpy
import threading
from groq import Groq
from config import CONFIGURATIONS, SYSTEM_PROMPT_BASE
from coco_interfaces.action import ProcessResponse
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from rclpy.lifecycle import LifecycleNode, LifecycleState, TransitionCallbackReturn

GROQ_API_KEY = os.getenv('GROQ_API_KEY')

class TextProcessor:
    @staticmethod
    def clean_text(text):
        """Clean text and split into sentences, similar to C++ implementation"""
        try:
            punct_pattern = r'[!¡?¿*,.:;()\[\]{}]'
            cleaned = re.sub(punct_pattern, ' ', text)
            
            cleaned = re.sub(r'\s+', ' ', cleaned)
            
            sentences = []
            sentence_pattern = r'([.!?])\s+'
            parts = re.split(sentence_pattern, cleaned)
            
            sentence = ""
            for i, part in enumerate(parts):
                sentence += part
                if re.match(r'[.!?]', part):
                    sentences.append(sentence.strip())
                    sentence = ""
            
            if sentence.strip():
                sentences.append(sentence.strip())
            
            return sentences if sentences else [cleaned.strip()]
        except Exception as e:
            return [text.strip()]

class LLMLifecycleNode(LifecycleNode):
    def __init__(self):
        super().__init__('llm_lifecycle_node')

        self.conversation_history = []
        
        self.action_server = None
        
        self.groq_client = Groq(api_key=GROQ_API_KEY)
        
        self.response_lock = threading.Lock()

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info('Configuring LLM Node')
        
        try:
            self.get_logger().info('Groq client initialized successfully')
            
            self.action_server = ActionServer(
                self,
                ProcessResponse,
                '/response_llama',
                self.execute_response_generation,
                goal_callback=self.handle_goal,
                cancel_callback=self.handle_cancel
            )
            
            self.get_logger().info('LLM Node configured successfully')
            return TransitionCallbackReturn.SUCCESS
            
        except Exception as e:
            self.get_logger().error(f'Failed to configure: {str(e)}')
            return TransitionCallbackReturn.FAILURE

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info('Activating LLM Node')
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info('Deactivating LLM Node')
        return TransitionCallbackReturn.SUCCESS

    def handle_goal(self, goal_request):
        self.get_logger().info('Received goal request')
        return GoalResponse.ACCEPT

    def handle_cancel(self, goal_handle):
        self.get_logger().info('Received cancel request')
        return CancelResponse.ACCEPT

    def execute_response_generation(self, goal_handle):
        self.get_logger().info('Executing response generation')
        
        feedback_msg = ProcessResponse.Feedback()
        result_msg = ProcessResponse.Result()

        if not self.groq_client:
            result_msg.completed = False
            goal_handle.abort()
            return result_msg

        try:
            self.conversation_history = []
            self.conversation_history.append({
                "role": "system",
                "content": SYSTEM_PROMPT_BASE
            })
            
            user_input = goal_handle.request.input_text
            self.conversation_history.append({
                "role": "user", 
                "content": user_input
            })

            completion = self.groq_client.chat.completions.create(
                model=CONFIGURATIONS['model'],
                messages=self.conversation_history,
                temperature=CONFIGURATIONS['temperature'],
                max_completion_tokens=CONFIGURATIONS['max_completion_tokens'],
                top_p=CONFIGURATIONS['top_p'],
                stream=True,
                stop=None,
            )

            buffer = ""
            full_response = ""

            self.get_logger().info("Model response:")

            for chunk in completion:
                if goal_handle.is_cancel_requested:
                    result_msg.completed = False
                    goal_handle.canceled()
                    return result_msg

                content = chunk.choices[0].delta.content
                if content:
                    buffer += content
                    full_response += content

                    sentence_end = 0
                    buffer_len = len(buffer)

                    while sentence_end < buffer_len:
                        next_end = -1
                        for punct in ['.', '!', '?']:
                            pos = buffer.find(punct, sentence_end)
                            if pos != -1 and (next_end == -1 or pos < next_end):
                                next_end = pos

                        if next_end == -1:
                            break

                        if next_end + 1 >= buffer_len or buffer[next_end + 1].isspace():
                            sentence_end = next_end + 1
                        else:
                            sentence_end = next_end + 1
                            continue

                        sentence = buffer[:sentence_end].strip()

                        if sentence:
                            clean_phrases = TextProcessor.clean_text(sentence)
                            if clean_phrases:
                                clean_phrase = clean_phrases[0]
                                feedback_msg.current_chunk = clean_phrase
                                feedback_msg.is_last_chunk = False
                                print(clean_phrase)
                                goal_handle.publish_feedback(feedback_msg)

                            buffer = buffer[sentence_end:]
                            buffer_len = len(buffer)
                            sentence_end = 0

            feedback_msg.is_last_chunk = True
            goal_handle.publish_feedback(feedback_msg)

            self.conversation_history.append({
                "role": "assistant",
                "content": full_response
            })

            result_msg.completed = True
            goal_handle.succeed(result_msg)
            return result_msg

        except Exception as e:
            self.get_logger().error(f'Error during response generation: {str(e)}')
            result_msg.completed = False
            goal_handle.abort()
            return result_msg


def main(args=None):
    rclpy.init(args=args)
    
    node = LLMLifecycleNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

