import os
import cv2
import numpy as np
import mediapipe as mp
from math import hypot
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import time
from collections import deque

# Suppress TensorFlow and MediaPipe warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class HandGestureController:
    def __init__(self):
        # Initialize MediaPipe with optimized settings
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,  # Balance between speed and accuracy
            min_detection_confidence=0.7,  # Slightly reduced for better detection in varying conditions
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils

        # Enhanced camera initialization with error handling
        self.initialize_camera()
        
        # Initialize Audio Control
        self.initialize_audio()
        
        # Optimization variables
        self.frame_processing_interval = 2  # Process every nth frame
        self.frame_counter = 0
        self.volume_history = deque(maxlen=5)  # Rolling average for volume smoothing
        self.last_volume_update = time.time()
        self.volume_update_interval = 0.05  # 50ms minimum between updates
        
        # Performance optimization flags
        self.enable_drawing = True  # Can be toggled for performance
        self.resolution_scale = 1.0  # Can be adjusted for performance
        
        # Initialize state variables
        self.landmark_indices = self.initialize_landmarks()
        self.smooth_volume = self.volume.GetMasterVolumeLevelScalar() * 100
        self.last_valid_volume = self.smooth_volume

    def initialize_camera(self):
        """Initialize camera with error handling and optimization"""
        max_attempts = 3
        for attempt in range(max_attempts):
            self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                # Optimize camera settings
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Reduced resolution
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency
                break
            if attempt == max_attempts - 1:
                raise RuntimeError("Failed to initialize webcam after multiple attempts")
            time.sleep(1)

    def initialize_audio(self):
        """Initialize audio control with error handling"""
        try:
            devices = AudioUtilities.GetSpeakers()
            interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            self.volume = cast(interface, POINTER(IAudioEndpointVolume))
            self.vol_range = self.volume.GetVolumeRange()
            self.min_vol = self.vol_range[0]
            self.max_vol = self.vol_range[1]
        except Exception as e:
            raise RuntimeError(f"Failed to initialize audio control: {e}")

    def initialize_landmarks(self):
        """Initialize hand landmarks mapping"""
        return {
            'THUMB_TIP': 4,
            'INDEX_TIP': 8,
            'MIDDLE_TIP': 12,
            'RING_TIP': 16,
            'PINKY_TIP': 20
        }

    def process_frame(self, frame):
        """Optimize frame processing"""
        # Resize frame for better performance
        if self.resolution_scale != 1.0:
            frame = cv2.resize(frame, None, fx=self.resolution_scale, 
                             fy=self.resolution_scale)
        
        # Convert to RGB more efficiently
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def get_hand_openness(self, hand_landmarks):
        """Optimized hand openness calculation"""
        thumb_tip = np.array([
            hand_landmarks.landmark[self.landmark_indices['THUMB_TIP']].x,
            hand_landmarks.landmark[self.landmark_indices['THUMB_TIP']].y
        ])
        
        # Calculate distances more efficiently
        distances = []
        for finger_tip in ['INDEX_TIP', 'MIDDLE_TIP', 'RING_TIP', 'PINKY_TIP']:
            tip = np.array([
                hand_landmarks.landmark[self.landmark_indices[finger_tip]].x,
                hand_landmarks.landmark[self.landmark_indices[finger_tip]].y
            ])
            distances.append(np.linalg.norm(thumb_tip - tip))
        
        return np.mean(distances)

    def update_volume(self, target_volume):
        """Update volume with improved smoothing and stability"""
        current_time = time.time()
        if current_time - self.last_volume_update < self.volume_update_interval:
            return self.last_valid_volume

        # Add to rolling average
        self.volume_history.append(target_volume)
        smooth_vol = int(np.mean(self.volume_history))
        
        # Update volume if significant change
        if abs(smooth_vol - self.last_valid_volume) >= 1:
            self.last_valid_volume = smooth_vol
            self.last_volume_update = current_time
            
            # Convert percentage to volume level
            vol_level = np.interp(smooth_vol, [0, 100], [self.min_vol, self.max_vol])
            self.volume.SetMasterVolumeLevel(vol_level, None)
        
        return self.last_valid_volume

    def run(self):
        print("Starting Hand Gesture Volume Control...")
        
        while True:
            success, frame = self.cap.read()
            if not success:
                print("Error: Failed to capture frame")
                time.sleep(0.1)
                continue

            # Process every nth frame for performance
            self.frame_counter += 1
            if self.frame_counter % self.frame_processing_interval != 0:
                continue

            # Optimize frame processing
            frame = cv2.flip(frame, 1)
            rgb_frame = self.process_frame(frame)
            results = self.hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                
                # Draw landmarks if enabled
                if self.enable_drawing:
                    self.mp_draw.draw_landmarks(frame, hand_landmarks, 
                                             self.mp_hands.HAND_CONNECTIONS)

                # Calculate and update volume
                hand_openness = self.get_hand_openness(hand_landmarks)
                target_volume = int(np.interp(hand_openness, [0.1, 0.3], [0, 100]))
                current_volume = self.update_volume(target_volume)
                
                # Display volume
                cv2.putText(frame, f"Volume: {current_volume}%", 
                          (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            cv2.imshow("Hand Gesture Volume Control", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cleanup()

    def cleanup(self):
        """Enhanced cleanup with error handling"""
        try:
            self.cap.release()
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Error during cleanup: {e}")

if __name__ == "__main__":
    try:
        controller = HandGestureController()
        print("\nOptimized Hand Gesture Control System initialized!")
        print("\nInstructions:")
        print("1. Volume Control:")
        print("   - Open hand wide: Maximum volume")
        print("   - Close hand: Minimum volume")
        print("\n2. Performance Features:")
        print("   - Optimized frame processing")
        print("   - Stable volume control")
        print("   - Enhanced hand detection")
        print("\nPress 'q' to quit")
        controller.run()
    except Exception as e:
        print(f"Fatal error: {e}")