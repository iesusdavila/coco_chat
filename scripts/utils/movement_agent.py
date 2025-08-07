from react_state import MovementState
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage

class MovementDetectionAgent:
    def __init__(self, llm: ChatGroq):
        self.llm = llm
        self.setup_graph()
    
    def setup_graph(self):
        graph = StateGraph(MovementState)
        graph.add_node("detect_movement", self.detect_movement_intent)
        graph.add_node("plan_movement", self.plan_joint_movement)
        graph.set_entry_point("detect_movement")
        graph.add_edge("detect_movement", "plan_movement")
        graph.add_edge("plan_movement", END)
        self.movement_graph = graph.compile()
    
    def detect_movement_intent(self, state: MovementState):
        """Detecta si hay intención de movimiento en el mensaje del usuario"""
        user_message = state["messages"][-2].content if state["messages"] else ""
        
        detection_prompt = f"""
        Analiza si el siguiente mensaje indica que el usuario quiere que un robot social realice un movimiento de sus extremidades o cabeza:
        
        Mensaje: "{user_message}"

        Los movimientos posibles son: saludar, afirmar, negar, aplaudir, señalar, girar_cabeza, girar_cuerpo, gesticular.
        En caso de haber otra intencion de movimiento ajusta el movimiento a uno de los anteriores. Es muy dificil que pida un movimiento que 
        ninguno asi que verifica bien el mensaje.
        
        Responde SOLO en el siguiente formato JSON:
        {{
            "movement_detected": true/false,
            "movement_type": tipo_de_movimiento (uno de los movimientos posibles o "ninguno")
        }}
        """
        
        response = self.llm.invoke([SystemMessage(content=detection_prompt)])
        
        try:
            import json
            result = json.loads(response.content.strip())
            movement_detected = result.get("movement_detected", False)
            movement_type = result.get("movement_type", "ninguno")
        except:
            movement_detected = False
            movement_type = "ninguno"
        
        return {
            "movement_detected": movement_detected,
            "movement_type": movement_type
        }
    
    def plan_joint_movement(self, state: MovementState):
        """Planifica qué joints mover según el tipo de movimiento detectado"""
        movement_type = state.get("movement_type", "ninguno")
        joints_to_move = {}
        
        if movement_type == "saludar":
            joints_to_move = {
                "joint_4": 1.0,
                "joint_5": 0.5,
                "joint_7": 0.8,
                "joint_8": 1.0
            }
        elif movement_type == "asentir":
            joints_to_move = {
                "joint_3": 0.5
            }
        elif movement_type == "negar":
            joints_to_move = {
                "joint_2": 0.7
            }
        elif movement_type == "aplaudir":
            joints_to_move = {
                "joint_4": 0.5,
                "joint_9": 0.5,
                "joint_8": 1.0,
                "joint_13": 1.0
            }
        elif movement_type == "girar_cabeza":
            joints_to_move = {
                "joint_2": 0.8
            }
        elif movement_type == "girar_cuerpo":
            joints_to_move = {
                "joint_1": 0.6
            }
        
        return {
            "joints_to_move": joints_to_move
        }
    
    def process_movement_intent(self, messages):
        """Procesa los mensajes y detecta intenciones de movimiento"""
        initial_state = {
            "messages": messages,
            "movement_detected": False,
            "movement_type": "ninguno",
            "joints_to_move": {}
        }
        
        result = self.movement_graph.invoke(initial_state)

        return result
