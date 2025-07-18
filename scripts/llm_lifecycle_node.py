#!/usr/bin/env python3

import re
import os
import uuid
import rclpy
import threading
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
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

        self.workflow = None
        self.app = None
        self.memory = None
        self.thread_id = str(uuid.uuid4())
        self.config = {"configurable": {"thread_id": self.thread_id}}
        
        self.action_server = None
        
        self.chat_model = ChatGroq(
            api_key=GROQ_API_KEY,
            model=CONFIGURATIONS['model'],
            temperature=CONFIGURATIONS['temperature'],
            max_tokens=CONFIGURATIONS['max_completion_tokens'],
            model_kwargs={"top_p": CONFIGURATIONS['top_p']}
        )
        
        self.response_lock = threading.Lock()
        
        self.user_configs = {}

    def get_or_create_user_config(self, user_id=None):
        if user_id is None:
            user_id = "default_user"
        
        if user_id not in self.user_configs:
            thread_id = str(uuid.uuid4())
            config = {"configurable": {"thread_id": thread_id}}
            self.user_configs[user_id] = config
            
            system_message = SystemMessage(content=SYSTEM_PROMPT_BASE)
            initial_state = {"messages": [system_message]}
            
            for event in self.app.stream(initial_state, config, stream_mode="values"):
                pass  # We just want to initialize the memory with system prompt
                
        return self.user_configs[user_id]

    def _setup_langgraph(self):
        self.workflow = StateGraph(state_schema=MessagesState)
        
        def call_model(state: MessagesState):
            response = self.chat_model.invoke(state["messages"])
            return {"messages": response}
        
        self.workflow.add_edge(START, "model")
        self.workflow.add_node("model", call_model)
        
        self.memory = MemorySaver()
        
        self.app = self.workflow.compile(checkpointer=self.memory)
        
        system_message = SystemMessage(content=SYSTEM_PROMPT_BASE)
        initial_state = {"messages": [system_message]}
        
        for event in self.app.stream(initial_state, self.config, stream_mode="values"):
            pass 

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info('Configuring LLM Node')
        
        try:
            self._setup_langgraph()
            self.get_logger().info('LangGraph workflow initialized successfully')
            
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

        if not self.app:
            result_msg.completed = False
            goal_handle.abort()
            return result_msg

        try:
            user_input = goal_handle.request.input_text
            
            user_config = self.get_or_create_user_config("default_user")
            
            current_state = self.app.get_state(user_config)
            messages = current_state.values.get("messages", [])
            
            self.get_logger().info(f"Current conversation has {len(messages)} messages in history")
            
            user_message = HumanMessage(content=user_input)
            full_conversation = messages + [user_message]
            
            self.get_logger().info(f"Sending {len(full_conversation)} messages to model for processing")
            
            buffer = ""
            full_response = ""

            self.get_logger().info("Model response:")

            for chunk in self.chat_model.stream(full_conversation):
                if goal_handle.is_cancel_requested:
                    result_msg.completed = False
                    goal_handle.canceled()
                    return result_msg

                content = chunk.content
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

            if buffer.strip():
                clean_phrases = TextProcessor.clean_text(buffer.strip())
                if clean_phrases:
                    clean_phrase = clean_phrases[0]
                    feedback_msg.current_chunk = clean_phrase
                    feedback_msg.is_last_chunk = False
                    print(clean_phrase)
                    goal_handle.publish_feedback(feedback_msg)

            feedback_msg.is_last_chunk = True
            goal_handle.publish_feedback(feedback_msg)

            ai_message = AIMessage(content=full_response)
            
            final_messages = messages + [user_message, ai_message]
            self.app.update_state(user_config, {"messages": final_messages})
            
            self.get_logger().info(f"Updated conversation state. Now has {len(final_messages)} messages in history")

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

