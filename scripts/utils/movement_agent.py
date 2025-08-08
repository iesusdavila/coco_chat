from react_state import MovementState
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage
from config import CONFIGURATIONS

class MovementDetectionAgent:
    def __init__(self):
        self.llm = ChatGroq(model="llama-3.1-8b-instant", streaming=True, 
                            max_tokens=CONFIGURATIONS['max_completion_tokens'], temperature=CONFIGURATIONS['temperature'],
                            model_kwargs={"top_p": CONFIGURATIONS['top_p']})
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
        user_message = state["messages"][-1].content if state["messages"] else ""
        
        detection_prompt = f"""
        Eres un experto en detectar intenciones de movimiento para un robot social. 
        Analiza si el siguiente mensaje indica que el usuario quiere que realice algún movimiento físico:
        
        Mensaje: "{user_message}"

        TIPOS DE MOVIMIENTO Y SUS VARIACIONES:
        
        1. SALUDAR: 
        - "saluda", "di hola", "haz un gesto de saludo", "mueve la mano", "agita la mano", "levanta la mano"
        
        2. MOVER_BRAZO_DERECHO:
        - "mueve el brazo derecho", "levanta el brazo derecho", "extiende el brazo derecho", "haz un gesto con el brazo derecho"

        3. MOVER_BRAZO_IZQUIERDO:
        - "mueve el brazo izquierdo", "levanta el brazo izquierdo", "extiende el brazo izquierdo", "haz un gesto con el brazo izquierdo"

        4. ABRIR_MANO_DERECHA:
        - "abre la mano derecha", "extiende tu mano derecha", "haz un gesto con la mano derecha", "abre la palma derecha"

        5. CERRAR_MANO_DERECHA:
        - "cierra la mano derecha", "haz un puño con la mano derecha", "junta tu mano derecha", "cierra la palma derecha"
        
        6. GIRAR_CABEZA:
        - "gira la cabeza", "mueve la cabeza", "voltea", "mira hacia", "gira tu cabeza"
        
        7. GIRAR_CUERPO:
        - "gira el cuerpo", "date la vuelta", "voltéate", "gira", "mueve el cuerpo"

        8. POSICION_ORIGINAL:
        - "vuelve a la posición original", "regresa a la posición inicial", "vuelve a estar quieto", "mantén la posición"

        9. ABRIR_MANO_IZQUIERDA:
        - "abre la mano izquierda", "extiende tu mano izquierda", "haz un gesto con la mano izquierda", "abre la palma izquierda"

        10. CERRAR_MANO_IZQUIERDA:
        - "cierra la mano izquierda", "haz un puño con la mano izquierda", "junta tu mano izquierda", "cierra la palma izquierda"

        INSTRUCCIONES:
        - Busca CUALQUIER palabra o frase que indique movimiento físico
        - Si encuentras sinónimos o variaciones de los movimientos listados, clasifícalos en la categoría apropiada
        - Si hay ambigüedad, elige el movimiento más probable basado en el contexto
        - Solo responde "ninguno" si definitivamente NO hay intención de movimiento físico
        
        EJEMPLOS:
        - "mueve tu cabeza" → girar_cabeza
        - "haz un gesto con tu brazo derecho" → mover_brazo_derecho
        - "levanta la mano derecha" → cerrar_mano_derecha
        - "¿puedes mover tu brazo izquierdo?" → mover_brazo_izquierdo
        - "dime tu nombre" → ninguno
        
        Responde SOLO en el siguiente formato JSON:
        {{
            "movement_detected": true/false,
            "movement_type": "mover_brazo_derecho/mover_brazo_izquierdo/abrir_mano_derecha/cerrar_mano_derecha/girar_cabeza/girar_cuerpo/posicion_original/abrir_mano_izquierda/cerrar_mano_izquierda/ninguno",
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
        
        if movement_type == "mover_brazo_derecho":
            joints_to_move = {
                "joint_4": 1.0,
                "joint_5": 0.5,
                "joint_6": 0.2,
                "joint_7": -0.4
            }
        elif movement_type == "mover_brazo_izquierdo":
            joints_to_move = {
                "joint_9": 1.0,
                "joint_10": 0.5,
                "joint_11": 0.2,
                "joint_12": -0.4
            }
        elif movement_type == "abrir_mano_derecha":
            joints_to_move = {
                "joint_8": 1.0
            }
        elif movement_type == "cerrar_mano_derecha":
            joints_to_move = {
                "joint_8": 0.0
            }
        elif movement_type == "abrir_mano_izquierda":
            joints_to_move = {
                "joint_13": 1.0
            }
        elif movement_type == "cerrar_mano_izquierda":
            joints_to_move = {
                "joint_13": 0.0
            }
        elif movement_type == "girar_cabeza":
            joints_to_move = {
                "joint_2": 0.8,
                "joint_3": 0.2
            }
        elif movement_type == "girar_cuerpo":
            joints_to_move = {
                "joint_1": 0.6
            }
        elif movement_type == "posicion_original":
            joints_to_move = {
                "joint_1": 0.0,
                "joint_2": 0.0,
                "joint_3": 0.0,
                "joint_4": 0.0,
                "joint_5": 0.0,
                "joint_6": 0.0,
                "joint_7": 0.0,
                "joint_8": 0.0,
                "joint_9": 0.0,
                "joint_10": 0.0,
                "joint_11": 0.0,
                "joint_12": 0.0,
                "joint_13": 0.0
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
