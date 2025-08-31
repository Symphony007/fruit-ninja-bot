import cv2
import time
import pyautogui
import math
import numpy as np
from src.vision.screen_capture import GameCapture
from src.vision.object_detection import find_targets
from src.vision.game_state import is_game_over
from src.vision.ml_detector import MLDetector
from src.vision.fruit_tracker import FruitTracker
from src.control.mouse_controller import MouseController
from src.utils.config import *

class FruitNinjaBot:
    def __init__(self):
        print("Initializing Fruit Ninja Bot - ULTRA MODE...")
        print("WARNING: This will actually move your mouse!")
        print("SAFETY: Move mouse to top-left corner to stop.")
        time.sleep(2)
        
        # Initialize components
        self.capture = GameCapture(CAPTURE_REGION, target_fps=60)
        self.mouse = MouseController()
        self.detector = MLDetector(ML_CONFIDENCE_THRESHOLD)
        self.tracker = FruitTracker()
        
        # Initialize ML detector
        if USE_ML_DETECTION:
            self.detector.initialize()
        
        # Game state variables
        self.last_swipe_time = 0
        self.frame_count = 0
        self.running = True
        self.total_swipes = 0
        self.total_hits = 0
        self.frame_h = CAPTURE_REGION[3] - CAPTURE_REGION[1]
        self.frame_w = CAPTURE_REGION[2] - CAPTURE_REGION[0]
        self.frames_skipped = 0
        
        # Performance tracking
        self.detection_times = []
        self.processing_times = []
        
        # Pomegranate detection
        self.pomegranate_candidates = {}
        self.rapid_mode = False
        self.rapid_end_time = 0

    def calculate_safety_score(self, fruit, bombs):
        """Calculate safety score for a fruit"""
        if not bombs:
            return 1.0
            
        fruit_x, fruit_y = fruit["center"]
        min_score = 1.0
        
        for bomb in bombs:
            bomb_x, bomb_y = bomb["center"]
            
            # Check horizontal path danger
            h_dist = abs(fruit_x - bomb_x)
            v_dist = abs(fruit_y - bomb_y)
            
            if h_dist < BOMB_HORIZONTAL_DANGER_DISTANCE and v_dist < BOMB_VERTICAL_DANGER_DISTANCE:
                return 0.0
            
            # Calculate general safety distance
            distance = math.sqrt(h_dist**2 + v_dist**2)
            safety = min(1.0, distance / BOMB_SAFETY_DISTANCE)
            min_score = min(min_score, safety)
        
        return min_score

    def is_swipe_safe(self, fruit, bombs):
        """Check if swipe path is clear of bombs"""
        fruit_x, fruit_y = fruit["center"]
        
        for bomb in bombs:
            bomb_x, bomb_y = bomb["center"]
            
            h_dist = abs(fruit_x - bomb_x)
            v_dist = abs(fruit_y - bomb_y)
            
            if h_dist < BOMB_HORIZONTAL_DANGER_DISTANCE and v_dist < BOMB_VERTICAL_DANGER_DISTANCE:
                return False
                
        return True

    def adjust_swipe_path(self, fruit_x, fruit_y, bombs):
        """Adjust swipe to avoid bombs"""
        start_x = fruit_x - SWIPE_LENGTH
        end_x = fruit_x + SWIPE_LENGTH
        
        for bomb in bombs:
            bomb_x, bomb_y = bomb["center"]
            
            if abs(fruit_y - bomb_y) < BOMB_VERTICAL_DANGER_DISTANCE:
                if bomb_x < fruit_x and abs(bomb_x - fruit_x) < 100:
                    start_x = max(start_x, bomb_x + 30)
                elif bomb_x > fruit_x and abs(bomb_x - fruit_x) < 100:
                    end_x = min(end_x, bomb_x - 30)
        
        # Ensure minimum swipe length
        if end_x - start_x < 50:
            return fruit_x - 50, fruit_x + 50
            
        return start_x, end_x

    def detect_pomegranate_mode(self, fruits, current_time):
        """Detect pomegranates and activate rapid mode"""
        if self.rapid_mode and current_time > self.rapid_end_time:
            self.rapid_mode = False
            print("\n🔁 Exiting rapid mode")
        
        for fruit in fruits:
            fruit_id = f"{fruit['center'][0]}_{fruit['center'][1]}"
            
            if fruit_id in self.pomegranate_candidates:
                first_seen, slices = self.pomegranate_candidates[fruit_id]
                
                if current_time - first_seen > POMEGRANATE_TIME_THRESHOLD and slices >= 2:
                    if not self.rapid_mode:
                        print("\n🎯 POMEGRANATE DETECTED! Entering rapid mode!")
                        self.rapid_mode = True
                        self.rapid_end_time = current_time + RAPID_MODE_DURATION
                    return True
        
        return self.rapid_mode

    def find_optimal_target(self, fruits, bombs, current_time):
        """Find the best fruit to target"""
        if not fruits or current_time - self.last_swipe_time < SWIPE_COOLDOWN:
            return None, 0.0
        
        best_target = None
        best_score = -1
        
        for fruit in fruits:
            # Check position (40-75% of screen height)
            fruit_y = fruit["center"][1]
            height_pct = fruit_y / self.frame_h
            if not (0.4 <= height_pct <= 0.75):
                continue
            
            # Check safety
            safety = self.calculate_safety_score(fruit, bombs)
            if safety < MIN_SAFETY_SCORE:
                continue
            
            if not self.is_swipe_safe(fruit, bombs):
                continue
            
            # Calculate final score
            confidence = fruit.get("confidence", 0.7)
            pos_score = 1.0 - abs(height_pct - 0.6)  # Best around 60%
            score = safety * confidence * pos_score
            
            if score > best_score:
                best_score = score
                best_target = fruit
        
        return best_target, best_score

    def track_pomegranate(self, fruit, current_time):
        """Track potential pomegranates"""
        fruit_id = f"{fruit['center'][0]}_{fruit['center'][1]}"
        
        if fruit_id not in self.pomegranate_candidates:
            self.pomegranate_candidates[fruit_id] = (current_time, 1)
        else:
            first_seen, slices = self.pomegranate_candidates[fruit_id]
            self.pomegranate_candidates[fruit_id] = (first_seen, slices + 1)
        
        # Clean old entries
        to_remove = [fid for fid, (seen, _) in self.pomegranate_candidates.items() 
                    if current_time - seen > 2.5]
        for fid in to_remove:
            del self.pomegranate_candidates[fid]

    def execute_swipe(self, fruit, bombs, current_time, is_rapid=False):
        """Execute a swipe with optimized parameters"""
        fruit_x, fruit_y = fruit["center"]
        
        # Adjust for bombs and convert to absolute coordinates
        start_x, end_x = self.adjust_swipe_path(fruit_x, fruit_y, bombs)
        
        abs_start_x = CAPTURE_REGION[0] + max(0, min(start_x, self.frame_w))
        abs_end_x = CAPTURE_REGION[0] + max(0, min(end_x, self.frame_w))
        abs_target_y = CAPTURE_REGION[1] + max(0, min(fruit_y + SWIPE_OFFSET_Y, self.frame_h))
        
        confidence = fruit.get("confidence", 0.7)
        mode = "RAPID" if is_rapid else "PRECISION"
        print(f"\n⚡ {mode} - Conf: {confidence:.2f}")
        
        # Execute the swipe with optimized timing
        self.mouse.perform_swipe(
            abs_start_x,
            abs_target_y,
            abs_end_x,
            abs_target_y,
            duration=SWIPE_DURATION * (0.7 if is_rapid else 1.0)
        )
        
        # Track for pomegranate detection
        self.track_pomegranate(fruit, current_time)
        
        self.total_swipes += 1
        self.last_swipe_time = current_time
        return 1

    def detect_objects_ml(self, frame):
        """ML-based object detection"""
        return self.detector.detect_objects(frame)

    def detect_objects_cv(self, frame):
        """CV-based fallback detection"""
        fruits = find_targets(frame, LOWER_RED, UPPER_RED, MIN_FRUIT_AREA)
        fruits.extend(find_targets(frame, LOWER_RED2, UPPER_RED2, MIN_FRUIT_AREA))
        bombs = find_targets(frame, LOWER_BOMB, UPPER_BOMB, MIN_BOMB_AREA)
        
        return ([{"type": "fruit", "center": f["center"], "confidence": 0.8} for f in fruits] +
                [{"type": "bomb", "center": b["center"], "confidence": 0.8} for b in bombs])

    def run(self):
        """Main bot execution loop"""
        try:
            frame_times = []
            detection_times = []
            
            while self.running:
                loop_start = time.time()
                current_time = time.time()
                
                # Failsafe check
                x, y = pyautogui.position()
                if x < 10 and y < 10:
                    raise pyautogui.FailSafeException("Failsafe triggered")
                
                # Capture frame
                frame = self.capture.get_frame()
                if frame is None:
                    continue
                
                # Frame skipping for performance
                self.frames_skipped += 1
                if self.frames_skipped < FRAME_SKIP:
                    continue
                self.frames_skipped = 0
                
                # Object detection
                detect_start = time.time()
                if USE_ML_DETECTION and self.detector.initialized:
                    detections = self.detect_objects_ml(frame)
                else:
                    detections = self.detect_objects_cv(frame)
                detection_times.append(time.time() - detect_start)
                
                # Separate fruits and bombs
                fruits = [d for d in detections if d["type"] == "fruit"]
                bombs = [d for d in detections if d["type"] == "bomb"]
                
                # Track fruits and predict positions
                tracked_fruits = self.tracker.update(fruits)
                targeted_fruits = [f.copy() for f in tracked_fruits]
                for f in targeted_fruits:
                    f["center"] = f["predicted_pos"]
                
                # Check for pomegranates
                rapid_mode = self.detect_pomegranate_mode(targeted_fruits, current_time)
                
                # Check game state periodically
                if len(targeted_fruits) == 0 and self.frame_count % 50 == 0:
                    if is_game_over(frame, GAME_OVER_TEMPLATE_PATH):
                        print("\n🎮 Game Over detected! Pausing...")
                        time.sleep(1.5)
                        continue
                
                # Find and execute swipe
                target, score = self.find_optimal_target(targeted_fruits, bombs, current_time)
                
                hits = 0
                if target and score > 0.4:  # Lower threshold for more aggressive play
                    hits = self.execute_swipe(target, bombs, current_time, rapid_mode)
                    self.total_hits += hits
                    action = f"{'RAPID' if rapid_mode else 'PRECISION'}"
                else:
                    action = "WAIT"
                
                # Performance monitoring
                frame_time = time.time() - loop_start
                frame_times.append(frame_time)
                if len(frame_times) > 20:
                    frame_times.pop(0)
                if len(detection_times) > 20:
                    detection_times.pop(0)
                
                fps = 1.0 / (sum(frame_times) / len(frame_times)) if frame_times else 0
                detect_fps = 1.0 / (sum(detection_times) / len(detection_times)) if detection_times else 0
                
                # Display performance info
                info = f"FPS={fps:.1f} D-FPS={detect_fps:.1f} Fruits={len(targeted_fruits)} Bombs={len(bombs)} Action={action}"
                print(info, end='\r', flush=True)
                
                self.frame_count += 1
                    
        except pyautogui.FailSafeException:
            print("\n✓ Failsafe triggered! Bot stopped.")
        except KeyboardInterrupt:
            print("\n✓ Bot stopped by user.")
        except Exception as e:
            print(f"\n✗ Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.capture.release()
            hit_rate = (self.total_hits / self.total_swipes * 100) if self.total_swipes > 0 else 0
            print(f"\n🏆 Bot stopped. Swipes: {self.total_swipes}, Hits: {self.total_hits}, Rate: {hit_rate:.1f}%")

if __name__ == "__main__":
    bot = FruitNinjaBot()
    bot.run()