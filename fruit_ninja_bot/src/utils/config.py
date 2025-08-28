import numpy as np
import os

# Get the base directory of the project (fruit_ninja_bot)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # This points to src/utils
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))  # Go up TWO levels to main folder

# Screen Capture Region (Your BlueStacks window coordinates) - UPDATED!
CAPTURE_REGION = (913, 171, 1885, 979)  # (left, top, right, bottom) - NEW COORDINATES

# Color Ranges in HSV format ([Hue, Saturation, Value]) - Fallback method
LOWER_RED = np.array([0, 150, 150])
UPPER_RED = np.array([10, 255, 255])
LOWER_RED2 = np.array([170, 150, 150])  # Red wraps around in HSV
UPPER_RED2 = np.array([180, 255, 255])
LOWER_BOMB = np.array([0, 0, 0])
UPPER_BOMB = np.array([180, 100, 100])

# Bot Behavior Settings
SWIPE_COOLDOWN = 0.7
BOMB_SAFETY_DISTANCE = 150  # pixels
MIN_FRUIT_AREA = 100
MIN_BOMB_AREA = 50
FRAME_SKIP = 2

# ML Detection Settings
ML_CONFIDENCE_THRESHOLD = 0.6
USE_ML_DETECTION = True  # Set to False to use color-based detection

# File Paths - UPDATED with correct relative paths from project root
GAME_OVER_TEMPLATE_PATH = os.path.join(PROJECT_ROOT, "data", "game_over_template.png")
MODEL_PATH = os.path.join(PROJECT_ROOT, "data", "models", "fruit_ninja_yolo.pt")
# If the custom model isn't ready, fall back to the base YOLO model in the root
YOLO_MODEL_PATH = os.path.join(PROJECT_ROOT, "yolov8n.pt")

# Print paths for debugging
print(f"PROJECT_ROOT: {PROJECT_ROOT}")
print(f"MODEL_PATH: {MODEL_PATH}")
print(f"GAME_OVER_TEMPLATE_PATH: {GAME_OVER_TEMPLATE_PATH}")