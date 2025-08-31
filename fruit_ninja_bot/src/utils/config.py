import numpy as np
import os

# Get the base directory of the project (fruit_ninja_bot)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # This points to src/utils
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))  # Go up TWO levels to main folder

# Screen Capture Region (Your BlueStacks window coordinates)
CAPTURE_REGION = (913, 171, 1885, 979)  # (left, top, right, bottom)

# Color Ranges in HSV format ([Hue, Saturation, Value]) - Fallback method
LOWER_RED = np.array([0, 150, 150])
UPPER_RED = np.array([10, 255, 255])
LOWER_RED2 = np.array([170, 150, 150])  # Red wraps around in HSV
UPPER_RED2 = np.array([180, 255, 255])
LOWER_BOMB = np.array([0, 0, 0])
UPPER_BOMB = np.array([180, 100, 100])

# Bot Behavior Settings
SWIPE_COOLDOWN = 0.15  # Reduced for faster response
BOMB_SAFETY_DISTANCE = 150  # pixels
MIN_FRUIT_AREA = 100
MIN_BOMB_AREA = 50
FRAME_SKIP = 1  # Process every 2nd frame for better tracking

# ML Detection Settings
ML_CONFIDENCE_THRESHOLD = 0.5  # Lower threshold for better detection
USE_ML_DETECTION = True

# Swipe Parameters - CRITICAL FOR REGISTRATION
SWIPE_LENGTH = 80  # Longer swipes for better registration
SWIPE_DURATION = 0.035  # Slightly slower for better registration
SWIPE_OFFSET_Y = -5  # Swipe slightly above center for better fruit cutting

# Safety Parameters
BOMB_HORIZONTAL_DANGER_DISTANCE = 80
BOMB_VERTICAL_DANGER_DISTANCE = 35
MIN_SAFETY_SCORE = 0.4  # Lower threshold for more aggressive play

# Timing Parameters
POMEGRANATE_TIME_THRESHOLD = 1.2
PREDICTION_TIME = 0.12  # 120ms prediction
RAPID_MODE_DURATION = 4.0

# File Paths
GAME_OVER_TEMPLATE_PATH = os.path.join(PROJECT_ROOT, "data", "game_over_template.png")
MODEL_PATH = os.path.join(PROJECT_ROOT, "data", "models", "fruit_ninja_yolo.pt")

# Print paths for debugging
print(f"PROJECT_ROOT: {PROJECT_ROOT}")
print(f"MODEL_PATH: {MODEL_PATH}")
print(f"GAME_OVER_TEMPLATE_PATH: {GAME_OVER_TEMPLATE_PATH}")