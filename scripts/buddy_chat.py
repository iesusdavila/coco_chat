#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from ament_index_python.packages import get_package_share_directory
from llama_cpp import Llama
import re
import threading
import queue
import tempfile
import wave
import time
import os
from piper import PiperVoice
from playsound import playsound
from vosk import Model, KaldiRecognizer
import json
import pyaudio


class ConversationModel:
    def __init__(self, model_path, context_size=2048, temperature=0.7):
        self.model = Llama(
            model_path=model_path,
            n_ctx=context_size,
            verbose=False,
            use_gpu=True,  
            n_gpu_layers=10,  
            threads=4,
            tensor_split=[1.0]
        )

        self.temperature = temperature
        self.conversation_history = []
        
    def generate_response(self, user_input, max_tokens=300):
        system_prompt = """
        Eres Leo, un asistente virtual amigable para niños que se encuentran en hospitales 
        y tienen entre 7 a 12 años. Sigue estas reglas:
        
        1. Respuestas:
        - Evita respuestas con caracteres especiales
        - Lenguae simple y claro
        2. Tonos: 
        - Entusiasta y positivo
        - Empático (ej: "¡Vaya, eso suena emocionante!")
        - Animador (ej: "¡Tú puedes!") 
        3. Seguridad:
        - Nunca solicites información personal
        - Redirige preguntas sensibles (ej: "Mejor pregúntale a tus papás sobre eso")
        - Corrige errores con amabilidad (ej: "En realidad... ¿sabías que...?")
        4. Interactividad:
        - Ofrece opciones múltiples (ej: "¿Quieres saber sobre animales o plantas?")
        - Incluye mini-juegos educativos (adivinanzas, trivia simple)
        - Usa formatos divertidos (lisdiálogos cortos)
        """
        
        self.conversation_history = [{"role": "system", "content": system_prompt}]

        self.conversation_history.append({"role": "user", "content": user_input})
        
        response_stream = self.model.create_chat_completion(
            messages=self.conversation_history,
            max_tokens=max_tokens,
            temperature=self.temperature,
            stream=True
        )
        
        return response_stream
    
    def add_assistant_response(self, response):
        """Añade la respuesta del asistente al historial de conversación"""
        self.conversation_history.append({"role": "assistant", "content": response})


class TextProcessor:
    """Clase para procesar y limpiar texto para TTS"""
    @staticmethod
    def clean_text(text):
        """Limpia el texto y prepara para síntesis de voz"""
        text = re.sub(r'[!¡?¿*,.:;()\[\]{}]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return re.split(r'(?<=[.!?])\s+', text)


class SpeechSystem:
    """Clase para manejar la síntesis y reconocimiento de voz"""
    def __init__(self, model_path, config_path, vosk_model_path, use_cuda=False, node=None):        
        self.voice = PiperVoice.load(
            model_path=model_path,
            config_path=config_path,
            use_cuda=use_cuda
        )
        self.audio_queue = queue.Queue()

        self.tts_running = threading.Event()
        self.tts_active = threading.Event()  
        self.audio_finished = threading.Event()
        self.stt_pause = threading.Event() 

        self.tts_thread = None
        
        self.vosk_model = Model(vosk_model_path)
        self.node = node    
        
    def start_tts_worker(self):
        """Inicia el hilo de trabajo para TTS"""
        self.tts_running.set()
        self.tts_thread = threading.Thread(target=self._tts_worker, daemon=True)
        self.tts_thread.start()
        
    def _tts_worker(self):
        """Hilo para procesamiento continuo de audio"""
        while self.tts_running.is_set():
            try:
                text_chunk = self.audio_queue.get(timeout=0.5)
                if text_chunk:
                    self.tts_active.set()
                    self.stt_pause.set()
                    
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as fp:
                        with wave.open(fp.name, 'wb') as wav_file:
                            wav_file.setnchannels(1)
                            wav_file.setsampwidth(2)
                            wav_file.setframerate(self.voice.config.sample_rate)
                            self.voice.synthesize(text_chunk, wav_file)
                        playsound(fp.name)
                    self.audio_queue.task_done()
                    if self.audio_queue.empty():
                        self.tts_active.clear()
                        self.stt_pause.clear()
                        self.audio_finished.set()
            except queue.Empty:
                continue
    
    def process_stream(self, response_stream):
        """Procesa el stream del modelo en tiempo real"""
        buffer = ""
        full_response = ""
        
        self.node.get_logger().info("Modelo: ")
        
        for token in response_stream:
            content = token['choices'][0]['delta'].get('content', '')
            buffer += content
            full_response += content

            print(content, end="", flush=True)
            
            if len(buffer) > 0 and content.endswith((' ', '.', ',', '!', '?')):
                for phrase in TextProcessor.clean_text(buffer):
                    if phrase:
                        self.audio_queue.put(phrase)
                buffer = ""
        
        print()

        return full_response
    
    def voice_to_text(self):
        """Convierte voz a texto usando Vosk"""
        self.node.get_logger().info("Preparado para escuchar... ")

        while self.tts_active.is_set() or self.stt_pause.is_set():
            time.sleep(0.05)
        
        p = pyaudio.PyAudio()

        stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8000)
        stream.start_stream()
        
        recognizer = KaldiRecognizer(self.vosk_model, 16000)
        
        self.node.get_logger().info("El modelo esta escuchando...")
        
        silence_limit = 20
        silence_threshold = 3000
        silent_chunks = 0
        silent_chunk_limit = int(silence_limit * 2)
        
        is_person_spoke = False
        
        while True:
            if self.stt_pause.is_set():
                time.sleep(0.1)
                continue
            data = stream.read(4000, exception_on_overflow=False)
            
            val_speech = max(abs(int.from_bytes(data[i:i+2], byteorder='little', signed=True)) for i in range(0, len(data), 2))
            is_speech = val_speech > silence_threshold
            
            if is_speech:
                silent_chunks = 0
                silent_chunk_limit = int(silent_chunk_limit / 2)
                is_person_spoke = True

            if not is_speech:
                silent_chunks += 1
                if silent_chunks > silent_chunk_limit:
                    silent_chunk_limit = int(silence_limit * 2)
                    if not is_person_spoke:
                        self.audio_queue.put("No puedo escucharte, por favor habla más fuerte.")
                        self.audio_finished.wait(timeout=30)
                    break
            
            if recognizer.AcceptWaveform(data):
                pass
            
            if is_speech and silent_chunks > silent_chunk_limit:
                break
                
        result_json = recognizer.FinalResult()
        result = json.loads(result_json)
        text = result.get("text", "").lower()

        stream.stop_stream()
        stream.close()
        p.terminate()
        
        if text:
            self.node.get_logger().info("Persona: " + text)
            return text
        else:
            self.node.get_logger().info("No se detectó ninguna entrada de voz.")
            return None
    
    def cleanup(self):
        """Limpia recursos del sistema de voz"""
        self.tts_running.clear()
        if self.tts_thread:
            self.tts_thread.join()
        self.audio_queue.queue.clear()

class VoiceAssistantNode(Node):
    """Nodo ROS2 para el asistente de voz"""
    def __init__(self):
        super().__init__('voice_assistant_node')
        
        pkg_share_dir = get_package_share_directory('buddy_chat')
        
        self.llm_model_path = os.path.join(pkg_share_dir, 'models', 'LLM/models--ggml-org--Meta-Llama-3.1-8B-Instruct-Q4_0-GGUF/snapshots/0aba27dd2f1c7f4941a94a5c59d80e0a256f9ff8', 'meta-llama-3.1-8b-instruct-q4_0.gguf')
        self.tts_model_path = os.path.join(pkg_share_dir, 'models', 'TTS', 'es_MX-claude-high.onnx')
        self.tts_config_path = os.path.join(pkg_share_dir, 'models', 'TTS', 'es_MX-claude-high.onnx.json')
        self.vosk_model_path = os.path.join(pkg_share_dir, 'models', 'STT', 'vosk-model-small-es-0.42')
        
        self.get_logger().info('Inicializando el modelo de conversación...')
        self.conversation_model = ConversationModel(self.llm_model_path)
        
        self.get_logger().info('Inicializando el sistema de voz...')
        self.speech_system = SpeechSystem(
            model_path=self.tts_model_path,
            config_path=self.tts_config_path,
            vosk_model_path=self.vosk_model_path,
            use_cuda=True,
            node=self
        )
        
        self.response_publisher = self.create_publisher(String, 'voice_assistant/response', 10)
        
        self.timer_active = True
        self.start_voice_mode()
        
        self.get_logger().info('Nodo de asistente de voz inicializado')
    
    def start_voice_mode(self):
        """Inicia el modo de voz cuando el timer se dispara"""
        if self.timer_active:
            self.timer_active = False
            self.get_logger().info('Iniciando modo de voz...')
            
            self.voice_thread = threading.Thread(target=self.run_voice_mode, daemon=True)
            self.voice_thread.start()
    
    def run_voice_mode(self):
        """Ejecuta el asistente en modo voz"""
        self.get_logger().info("Hola soy Leo, tu gran amigo. ¿En qué puedo ayudarte?")
        self.speech_system.start_tts_worker()

        self.speech_system.audio_queue.put("Hola soy Leo, tu gran amigo. ¿En qué puedo ayudarte?")
        self.speech_system.audio_finished.wait(timeout=10)
        
        try:
            while rclpy.ok():
                while not self.speech_system.audio_queue.empty():
                    time.sleep(0.1)
            
                self.get_logger().info("Escuchando...")
                user_input = self.speech_system.voice_to_text()
                
                if user_input is None:
                    continue
                
                if 'terminar' in user_input:
                    self.speech_system.audio_queue.put("Hasta luego, que tengas un buen día.")
                    term_msg = String()
                    term_msg.data = "terminar"
                    self.response_publisher.publish(term_msg)
                    self.speech_system.audio_finished.wait(timeout=30)
                    rclpy.shutdown()
                    break
                    
                response_stream = self.conversation_model.generate_response(user_input)
                response = self.speech_system.process_stream(response_stream)
                self.conversation_model.add_assistant_response(response)
                
                response_msg = String()
                response_msg.data = response
                self.response_publisher.publish(response_msg)
                
                self.speech_system.audio_finished.wait(timeout=30)
        except Exception as e:
            self.get_logger().error(f'Error en el modo de voz: {str(e)}')
        finally:
            self.speech_system.cleanup()


def main(args=None):
    rclpy.init(args=args)
    
    node = VoiceAssistantNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Nodo interrumpido por el usuario')
    except Exception as e:
        node.get_logger().error(f'Error en el nodo: {str(e)}')
    finally:
        node.speech_system.cleanup()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()