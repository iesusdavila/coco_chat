#!/usr/bin/env python3

import rclpy
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn 
from std_msgs.msg import Bool
from coco_interfaces.msg import PersonResponse
from ament_index_python.packages import get_package_share_directory
from vosk import Model, KaldiRecognizer, SetLogLevel
import pyaudio
import json
import os
import threading
from contextlib import contextmanager
from ctypes import CFUNCTYPE, c_char_p, c_int, cdll

ERROR_HANDLER_FUNC = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)

def py_error_handler(filename, line, function, err, fmt):
    pass

c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)

@contextmanager
def noalsaerr():
    asound = cdll.LoadLibrary('libasound.so')
    asound.snd_lib_error_set_handler(c_error_handler)
    yield
    asound.snd_lib_error_set_handler(None)


SetLogLevel(-1)

class STTLifecycleNode(LifecycleNode):
    def __init__(self):
        super().__init__('stt_lifecycle_node')
        
        pkg_share_dir = get_package_share_directory('coco_chat')
        self.vosk_model_path = os.path.join(pkg_share_dir, 'models', 'STT', 'vosk-model-es-0.42')
        
        self.response_publisher = self.create_publisher(PersonResponse, '/response_person', 10)
        self.stt_status_publisher = self.create_publisher(Bool, '/stt_terminado', 10)
        
        self.vosk_model = None
        self.recognition_thread = None
        self.is_recognizing = False
        
    def on_configure(self, state):
        self.get_logger().info('Configuring STT Node')
        
        try:
            self.vosk_model = Model(self.vosk_model_path)
            return TransitionCallbackReturn.SUCCESS
        except Exception as e:
            return TransitionCallbackReturn.FAILURE
    
    def on_activate(self, state):
        self.get_logger().info('Activating STT Node')
        
        self.is_recognizing = True
        
        self.recognition_thread = threading.Thread(target=self._recognize_speech)
        self.recognition_thread.start()
        
        return TransitionCallbackReturn.SUCCESS
    
    def on_deactivate(self, state):
        self.get_logger().info('Deactivating STT Node')
        
        self.is_recognizing = False
        if self.recognition_thread:
            self.recognition_thread.join()
        
        return TransitionCallbackReturn.SUCCESS
    
    def _recognize_speech(self):
        """Speech recognition method"""
        with noalsaerr():
            p = pyaudio.PyAudio()
            stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8000)
        recognizer = KaldiRecognizer(self.vosk_model, 16000)
        
        try:
            while self.is_recognizing:
                data = stream.read(4000, exception_on_overflow=False)
                
                if recognizer.AcceptWaveform(data):
                    result_json = recognizer.Result()
                    result = json.loads(result_json)
                    text = result.get("text", "").lower().strip()
                    
                    if text:
                        response_msg = PersonResponse()
                        response_msg.text = text
                        response_msg.timestamp = self.get_clock().now().to_msg()
                        
                        self.response_publisher.publish(response_msg)
                        self.get_logger().info(f"Person: {text}")

                        status_msg = Bool()
                        status_msg.data = True
                        self.stt_status_publisher.publish(status_msg)

                        self.is_recognizing = False
        
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()

def main(args=None):
    rclpy.init(args=args)
    node = STTLifecycleNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()