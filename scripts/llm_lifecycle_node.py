#!/usr/bin/env python3

import rclpy
import threading
from react_state import ChatState
from text_processor import TextProcessor
from movement_agent import MovementDetectionAgent
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from config import CONFIGURATIONS, SYSTEM_PROMPT_BASE
from coco_interfaces.action import ProcessResponse
from rclpy.action import ActionServer, ActionClient, GoalResponse, CancelResponse
from rclpy.lifecycle import LifecycleNode, LifecycleState, TransitionCallbackReturn
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from builtin_interfaces.msg import Duration

class LLMLifecycleNode(LifecycleNode):
    def __init__(self):
        super().__init__('llm_lifecycle_node')

        self.memory = MemorySaver()
        self.llm = ChatGroq(model=CONFIGURATIONS['model'], streaming=True, 
                            max_tokens=CONFIGURATIONS['max_completion_tokens'], temperature=CONFIGURATIONS['temperature'],
                            model_kwargs={"top_p": CONFIGURATIONS['top_p']})
        
        self.movement_agent = MovementDetectionAgent(self.llm)
        
        self.graph = StateGraph(ChatState)
        self.graph.add_node("chatbot", self.chatbot)
        self.graph.add_node("check_movement", self.check_movement_intent)
        self.graph.add_edge("chatbot", "check_movement")
        self.graph.add_edge("check_movement", END)
        self.graph.set_entry_point("chatbot")
        
        self.app = self.graph.compile(checkpointer=self.memory)
        
        self.config = {"configurable": {"thread_id": 1}}
        
        self.joint_action_client = ActionClient(self, FollowJointTrajectory, '/joint_trajectory_controller/follow_joint_trajectory')
        
        self.action_server = None
        
        self.response_lock = threading.Lock()

    def chatbot(self, state: ChatState):
        return {
            "messages": [self.llm.invoke(state["messages"])]
        }
    
    def check_movement_intent(self, state: ChatState):
        """Verifica si hay intención de movimiento y ejecuta acciones del robot"""
        movement_result = self.movement_agent.process_movement_intent(state["messages"])

        self.get_logger().info(f"Verifying movement intent: {movement_result}")
        
        if movement_result["movement_detected"]:
            self.get_logger().info(f"Movement detected: {movement_result['movement_type']}")
            self.execute_robot_movement(movement_result["joints_to_move"])
            
            state["movement_intent"] = movement_result
            state["robot_action_required"] = True
        else:
            self.get_logger().info("Don't need to move the robot")
            state["robot_action_required"] = False
            
        return state
    
    def execute_robot_movement(self, joints_to_move: dict):
        """Ejecuta el movimiento del robot usando action /joint_trajectory_controller/follow_joint_trajectory"""
        if not joints_to_move:
            return
            
        if not self.joint_action_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error("Action server /joint_trajectory_controller/follow_joint_trajectory no disponible")
            return
            
        goal_msg = FollowJointTrajectory.Goal()
        
        joint_names = [f"joint_{i}" for i in range(1, 14)]
        goal_msg.trajectory.joint_names = joint_names
        
        point = JointTrajectoryPoint()
        
        point.positions = [0.0] * 13
        
        for joint_name, position in joints_to_move.items():
            if joint_name in joint_names:
                joint_index = joint_names.index(joint_name)
                point.positions[joint_index] = position
        
        point.time_from_start = Duration(sec=3, nanosec=0)
        
        goal_msg.trajectory.points = [point]
        
        future = self.joint_action_client.send_goal_async(goal_msg)
        
        self.get_logger().info(f"Comando de movimiento enviado: {joints_to_move}")
    
    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info('Configuring LLM Node')
        
        try:
            self.get_logger().info('LangGraph with ChatGroq initialized successfully')
            
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

        try:
            user_input = goal_handle.request.input_text
            
            messages_to_send = [
                SystemMessage(content=SYSTEM_PROMPT_BASE),
                HumanMessage(content=user_input)
            ]
            
            result = self.app.invoke({
                "messages": messages_to_send,
                "movement_intent": None,
                "robot_action_required": False
            }, config=self.config)
            
            ai_response = result["messages"][-1].content

            buffer = ""

            self.get_logger().info("Model response:")

            for char in ai_response:
                if goal_handle.is_cancel_requested:
                    result_msg.completed = False
                    goal_handle.canceled()
                    return result_msg

                buffer += char

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
                            self.get_logger().info(clean_phrase)
                            goal_handle.publish_feedback(feedback_msg)

                        buffer = buffer[sentence_end:]
                        buffer_len = len(buffer)
                        sentence_end = 0

            feedback_msg.is_last_chunk = True
            goal_handle.publish_feedback(feedback_msg)

            result_msg.completed = True
            goal_handle.succeed()
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
