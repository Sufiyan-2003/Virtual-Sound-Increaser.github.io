🎛️ Hand Gesture Volume Control System

A real-time, AI-powered hand gesture recognition system that lets users control their system volume using only hand movements — no physical contact required.
Built using MediaPipe, OpenCV, and PyCAW, this project provides a smarter, intuitive, and touchless way to interact with your device.

🚀 Features
✅ 1. Touchless Volume Control

Control system volume by simply opening or closing your hand:

Open hand → Increase volume

Closed hand → Decrease volume

✅ 2. Optimized Hand Tracking (MediaPipe Hands)

High-accuracy hand landmark detection

Works in real-time

Tracks up to 21 key landmarks on the hand

✅ 3. Intelligent Volume Smoothing

Prevents sudden spikes in audio levels

Uses rolling average (deque) to stabilize output

Updates only when meaningful changes occur

✅ 4. Performance Optimizations

Processes every nth frame for smoother output

Frame resizing + reduced resolution options

Adjustable drawing and processing settings

FPS-optimized camera configuration

✅ 5. Robust Error Handling

Webcam initialization retries

Safe cleanup on exit

Audio control exception handling

🧠 How It Works
🔹 Hand Openness Detection

The system measures the distance between:

Thumb tip → Other fingers’ tips
The average distance determines how open the hand is, which is mapped to a volume percentage (0–100%).

🔹 Volume Mapping
Hand openness 0.1 → Volume 0%
Hand openness 0.3 → Volume 100%

🔹 Real-Time Adjustments

Volume update frequency regulated to 50ms

Uses PyCAW to directly control system audio

🛠️ Technologies Used
Technology	Purpose
Python	Main programming language
OpenCV	Camera input & image processing
MediaPipe Hands	Real-time hand tracking
NumPy	Distance calculations
PyCAW	Windows audio control
Deque	Rolling average for smoothing
📸 Live Demo Window

Shows real-time camera feed

Displays hand landmarks

Shows dynamically updated volume percentage

Press 'q' anytime to exit.

🏁 Running the Project
Install Dependencies
pip install opencv-python mediapipe comtypes pycaw numpy

Run the Program
python hand_gesture_volume_control.py

📌 System Requirements

Windows OS (PyCAW requirement)

Webcam (built-in or external)

Python 3.7+

📤 Features Included in Code

Webcam optimization

RGB frame conversion speed-ups

MediaPipe landmark drawing toggle

Volume interpolation and smoothing

Stable gesture detection

Error-safe startup and shutdown

🧹 Cleanup & Exit

The script safely:

Releases webcam

Closes all OpenCV windows

Simply press ‘q’ to quit.
