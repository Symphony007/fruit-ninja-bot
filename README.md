# 🍉 Fruit Ninja Bot

An AI-powered automation bot for the classic Fruit Ninja game, built with Python and YOLOv8. This bot uses real-time screen capture, computer vision, and precise input simulation to identify fruits and swipe them off the screen.

## 🗂️ Project Structure

```
fruit_ninja_bot/
├── data/                           # Dataset and Model Files
│   ├── models/
│   │   └── fruit_ninja_yolo/       # Custom-trained YOLOv5 model weights and results
│   ├── yolo_dataset/               # YOLO-formatted dataset (images & labels)
│   │   ├── train/
│   │   ├── valid/
│   │   ├── test/
│   │   └── data.yaml               # Dataset configuration file
│   └── game_over_template.png      # Template image for detecting the "Game Over" screen
├── src/                            # Core Source Code
│   ├── vision/                     # Everything related to computer vision
│   │   ├── screen_capture.py       # Captures and processes screen regions
│   │   ├── object_detection.py     # Base class for object detection
│   │   ├── ml_detector.py          # YOLOv8 model integration and inference
│   │   ├── fruit_tracker.py        # Tracks fruit trajectories and properties
│   │   └── game_state.py           # Determines the current state of the game (e.g., Game Over)
│   ├── control/                    # Input simulation
│   │   └── mouse_controller.py     # Controls the mouse to swipe fruits
│   ├── strategies/                 # Different algorithms for playing
│   │   └── ml_strategy.py          # Strategy using ML model predictions for decision making
│   └── utils/
│       └── config.py               # Centralized configuration settings
├── calibrate_region.py             # Helper script to select the game window area
├── debug_live.py                   # Script to test vision pipeline with live preview
├── real_bot.py                     # main script
├── requirements.txt                # Python dependencies
└── README.md
```

## ⚙️ Prerequisites

Before you begin, ensure you have the following installed:

- Python 3.8 or higher
- The game Fruit Ninja (e.g., from Steam or Bluestacks )
- An NVIDIA GPU is recommended for best performance (CUDA support), but the bot will also run on CPU.

## 🚀 Installation & Setup

Follow these steps to get the bot running on your system.

### 1. Clone the Repository

```bash
git clone https://github.com/Symphony007/fruit-ninja-bot
cd fruit-ninja-bot
```

### 2. Create a Virtual Environment (Highly Recommended)

```bash
# For Windows
python -m venv venv
.\venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install libraries like torch, torchvision, opencv-python, numpy, pyautogui, and mss.

### 4. Calibrate the Screen Capture Region (CRITICAL STEP)

The bot needs to know where the game is on your screen. You must run this script and follow the instructions.

```bash
python calibrate_region.py
```

A transparent window will appear. Move it and resize it so it perfectly covers the Fruit Ninja game area (where fruits appear and are sliced).

Press Enter to confirm the selection. The coordinates will be saved automatically to `src/utils/config.py`.

### 5. (Optional) Test the Vision Pipeline

Before running the full bot, verify that it can see the game correctly.

```bash
python debug_live.py
```

This will open a window showing a live feed of what the bot sees, with object detection and fruit tracking overlays. This is crucial for debugging.

## 🎮 How to Use

1. Start the Fruit Ninja game. Ensure it's in the foreground.

2. Run the bot:
   ```bash
   python real_bot.py
   ```

3. The bot will start processing your screen and playing the game. It will automatically detect the "Game Over" screen and stop, or you can stop it manually by moving your mouse to the top-left corner of the screen (pyautogui failsafe).

## 🔧 Configuration

You can fine-tune the bot's behavior by modifying `src/utils/config.py`. Key settings include:

- **GAME_REGION**: The screen coordinates of the game (set automatically by calibrate_region.py).
- **SWIPE_DELAY**: The delay between swipes. Adjust if the bot is too fast/slow.
- **CONFIDENCE_THRESHOLD**: The minimum confidence score for the ML model to detect a fruit. Increase if there are too many false positives.
- **GAME_OVER_THRESHOLD**: The similarity threshold for detecting the "Game Over" screen.

## 🧠 How It Works

1. **Screen Capture**: The `screen_capture.py` module continuously grabs the predefined region of your screen.

2. **Object Detection**: The captured image is passed to the YOLOv5 model (`ml_detector.py`) which identifies fruits and their bounding boxes.

3. **Tracking & Prediction**: The `fruit_tracker.py` module tracks each fruit across frames, calculating their velocity and predicted future position.

4. **Strategy & Decision**: The `ml_strategy.py` decides which fruit to target and the optimal swipe trajectory to slice it.

5. **Input Simulation**: The `mouse_controller.py` executes the swipe by simulating mouse movements.

6. **State Management**: The `game_state.py` module constantly checks for the "Game Over" screen to know when to stop.

## 🤖 Training Your Own Model (For Developers)

The provided model (`data/models/fruit_ninja_yolo/`) is already trained. If you want to improve it or add new fruit types:

1. Collect more screenshots and annotate them with a tool like LabelImg.

2. Organize the images and labels into the `data/yolo_dataset/train/` and `valid/` folders.

3. Update `data/yolo_dataset/data.yaml` with the correct paths and class names.

4. Run the YOLOv5 training command:
   ```bash
   python train.py --img 640 --batch 16 --epochs 50 --data data/yolo_dataset/data.yaml --weights yolov5s.pt
   ```

5. Place the new `best.pt` model file in the `data/models/` directory and update the model path in `config.py`.

## ⚠️ Disclaimer

This bot is intended for educational and personal use only. Using automation software in games may violate their Terms of Service. Use at your own risk.
