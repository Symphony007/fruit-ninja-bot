import cv2
import time
import pyautogui
import math
from src.vision.screen_capture import GameCapture
from src.vision.object_detection import find_targets
from src.vision.game_state import is_game_over
from src.vision.ml_detector import MLDetector
from src.vision.fruit_tracker import FruitTracker
from src.control.mouse_controller import MouseController
from src.utils.config import *

class FruitNinjaBot:
    def __init__(self):
        print("Initializing Fruit Ninja Bot - INTELLIGENT MODE...")
        print("WARNING: This will actually move your mouse!")
        print("SAFETY: Move mouse to top-left corner to stop.")
        time.sleep(2)
        
        # Initialize components
        self.game_capture = GameCapture(CAPTURE_REGION, target_fps=60)
        self.mouse_controller = MouseController()
        self.ml_detector = MLDetector(ML_CONFIDENCE_THRESHOLD)
        self.fruit_tracker = FruitTracker()
        
        # Initialize ML detector
        if USE_ML_DETECTION:
            self.ml_detector.initialize()
        
        # PROVEN SWIPE SETTINGS (from working version)
        self.swipe_length = 60
        self.swipe_duration = 0.020
        self.cooldown = 0.1
        
        # Game state variables
        self.last_swipe_time = 0
        self.frame_counter = 0
        self.running = True
        self.total_swipes = 0
        self.total_fruits_cut = 0
        self.frame_height = CAPTURE_REGION[3] - CAPTURE_REGION[1]
        self.frame_width = CAPTURE_REGION[2] - CAPTURE_REGION[0]
        
        # Pomegranate detection
        self.potential_pomegranates = {}  # {fruit_id: (first_seen_time, slice_count)}
        self.rapid_mode = False
        self.rapid_mode_end_time = 0
        
    def print_timing_info(self, fps, fruits_count, bombs_count, action):
        """Print timing information"""
        timing_str = f"FPS={fps:.1f} Fruits={fruits_count} Bombs={bombs_count} Action={action}"
        print(timing_str, end='\r', flush=True)
    
    def calculate_safety_score(self, fruit, bombs):
        """Calculate safety score with bomb proximity check"""
        if not bombs:
            return 1.0
            
        fruit_x, fruit_y = fruit["center"]
        min_safety = 1.0
        
        for bomb in bombs:
            bomb_x, bomb_y = bomb["center"]
            distance = math.sqrt((fruit_x - bomb_x)**2 + (fruit_y - bomb_y)**2)
            
            # Critical: Check if bomb is in the horizontal swipe path
            horizontal_distance = abs(fruit_x - bomb_x)
            vertical_difference = abs(fruit_y - bomb_y)
            
            # If bomb is directly in horizontal path, very dangerous
            if horizontal_distance < 60 and vertical_difference < 25:
                return 0.0
            
            distance_safety = min(1.0, distance / 100.0)
            min_safety = min(min_safety, distance_safety)
        
        return min_safety
    
    def is_swipe_safe(self, fruit, bombs):
        """Check if horizontal swipe would hit any bombs"""
        fruit_x, fruit_y = fruit["center"]
        
        for bomb in bombs:
            bomb_x, bomb_y = bomb["center"]
            
            # Check if bomb is in the horizontal swipe path
            horizontal_distance = abs(fruit_x - bomb_x)
            vertical_difference = abs(fruit_y - bomb_y)
            
            if horizontal_distance < 70 and vertical_difference < 30:
                return False
        
        return True
    
    def adjust_swipe_for_bombs(self, fruit_x, fruit_y, bombs):
        """Adjust swipe to avoid bombs while maintaining effectiveness"""
        base_start = fruit_x - self.swipe_length
        base_end = fruit_x + self.swipe_length
        
        for bomb in bombs:
            bomb_x, bomb_y = bomb["center"]
            
            # Only adjust for bombs in horizontal path
            if abs(fruit_y - bomb_y) < 30:
                # Bomb is to the left of fruit
                if bomb_x < fruit_x and abs(bomb_x - fruit_x) < 80:
                    base_start = max(base_start, bomb_x + 25)  # Give wider berth
                
                # Bomb is to the right of fruit
                elif bomb_x > fruit_x and abs(bomb_x - fruit_x) < 80:
                    base_end = min(base_end, bomb_x - 25)
        
        # Ensure we still have a meaningful swipe length
        if base_end - base_start < 40:
            return fruit_x - 40, fruit_x + 40
        
        return base_start, base_end
    
    def check_pomegranate_behavior(self, fruits, current_time):
        """Your brilliant idea: detect pomegranates by their behavior"""
        if self.rapid_mode and current_time > self.rapid_mode_end_time:
            self.rapid_mode = False
            print("\n🔁 Exiting rapid mode")
        
        # Check all fruits for pomegranate behavior
        for fruit in fruits:
            fruit_id = f"{fruit['center'][0]}_{fruit['center'][1]}"
            
            # If fruit stays on screen for a while and gets swiped multiple times
            if fruit_id in self.potential_pomegranates:
                first_seen, slice_count = self.potential_pomegranates[fruit_id]
                
                # If fruit persists for more than 1 second, it's likely a pomegranate
                if current_time - first_seen > 1.0 and slice_count >= 2:
                    if not self.rapid_mode:
                        print("\n🎯 POMEGRANATE DETECTED! Entering rapid mode!")
                        self.rapid_mode = True
                        self.rapid_mode_end_time = current_time + 3.0  # Rapid mode for 3 seconds
                    return True
        
        return self.rapid_mode
    
    def find_best_swipe_target(self, fruits, bombs, current_time):
        """
        Find the single best fruit to swipe with optimal timing.
        """
        if not fruits or current_time - self.last_swipe_time < self.cooldown:
            return None, 0.0
        
        best_target = None
        best_score = -1
        
        for fruit in fruits:
            # Check if fruit is in optimal height range (40-70%)
            fruit_y = fruit["center"][1]
            height_percentage = fruit_y / self.frame_height
            if not (0.4 <= height_percentage <= 0.7):
                continue
            
            # Check safety
            safety_score = self.calculate_safety_score(fruit, bombs)
            if safety_score < 0.6:
                continue
            
            if not self.is_swipe_safe(fruit, bombs):
                continue
            
            # Score based on safety, confidence, and position
            confidence = fruit.get("confidence", 1.0)
            position_score = 1.0 - abs(height_percentage - 0.55)  # Best around 55%
            total_score = safety_score * confidence * position_score
            
            if total_score > best_score:
                best_score = total_score
                best_target = fruit
        
        return best_target, best_score
    
    def track_pomegranate_candidate(self, fruit, current_time):
        """Track fruits that might be pomegranates"""
        fruit_id = f"{fruit['center'][0]}_{fruit['center'][1]}"
        
        if fruit_id not in self.potential_pomegranates:
            # New fruit - start tracking
            self.potential_pomegranates[fruit_id] = (current_time, 1)
        else:
            # Existing fruit - increment slice count
            first_seen, slice_count = self.potential_pomegranates[fruit_id]
            self.potential_pomegranates[fruit_id] = (first_seen, slice_count + 1)
        
        # Clean up old entries
        to_remove = []
        for fid, (seen_time, _) in self.potential_pomegranates.items():
            if current_time - seen_time > 2.0:  # Remove after 2 seconds
                to_remove.append(fid)
        
        for fid in to_remove:
            del self.potential_pomegranates[fid]
    
    def execute_swipe(self, fruit, bombs, current_time, is_rapid_mode=False):
        """Execute a single precise swipe"""
        fruit_x, fruit_y = fruit["center"]
        
        # Adjust swipe to avoid bombs
        start_x, end_x = self.adjust_swipe_for_bombs(fruit_x, fruit_y, bombs)
        
        # Convert to absolute coordinates
        abs_start_x = CAPTURE_REGION[0] + max(0, min(start_x, self.frame_width))
        abs_end_x = CAPTURE_REGION[0] + max(0, min(end_x, self.frame_width))
        abs_target_y = CAPTURE_REGION[1] + max(0, min(fruit_y, self.frame_height))
        
        confidence = fruit.get("confidence", 1.0)
        mode = "RAPID" if is_rapid_mode else "PRECISION"
        print(f"\n⚡ {mode} - Conf: {confidence:.2f}")
        
        # Execute swipe
        self.mouse_controller.perform_swipe(
            abs_start_x,
            abs_target_y,
            abs_end_x,
            abs_target_y,
            duration=self.swipe_duration * (0.8 if is_rapid_mode else 1.0)
        )
        
        # Track for pomegranate detection
        self.track_pomegranate_candidate(fruit, current_time)
        
        self.total_swipes += 1
        self.last_swipe_time = current_time
        return 1

    def detect_objects_ml(self, frame):
        return self.ml_detector.detect_objects(frame)

    def detect_objects_cv(self, frame):
        fruits = find_targets(frame, LOWER_RED, UPPER_RED, MIN_FRUIT_AREA)
        fruits2 = find_targets(frame, LOWER_RED2, UPPER_RED2, MIN_FRUIT_AREA)
        fruits.extend(fruits2)
        
        bombs = find_targets(frame, LOWER_BOMB, UPPER_BOMB, MIN_BOMB_AREA)
        
        ml_fruits = [{"type": "fruit", "center": f["center"], "confidence": 1.0} for f in fruits]
        ml_bombs = [{"type": "bomb", "center": b["center"], "confidence": 1.0} for b in bombs]
        
        return ml_fruits + ml_bombs

    def separate_fruits_bombs(self, detections):
        fruits = [d for d in detections if d["type"] == "fruit"]
        bombs = [d for d in detections if d["type"] == "bomb"]
        return fruits, bombs

    def run(self):
        """Main bot loop - intelligent with pomegranate detection"""
        try:
            frame_times = []
            
            while self.running:
                loop_start = time.time()
                current_time = time.time()
                
                # Check failsafe
                x, y = pyautogui.position()
                if x < 10 and y < 10:
                    raise pyautogui.FailSafeException("Failsafe triggered by user")
                
                # Capture frame
                frame = self.game_capture.get_frame()
                if frame is None:
                    continue
                
                # Detect objects
                if USE_ML_DETECTION and self.ml_detector.initialized:
                    detections = self.detect_objects_ml(frame)
                else:
                    detections = self.detect_objects_cv(frame)
                
                # Separate fruits and bombs
                fruits, bombs = self.separate_fruits_bombs(detections)
                
                # Update fruit tracks
                fruits_with_predictions = self.fruit_tracker.update(fruits)
                targeted_fruits = [f.copy() for f in fruits_with_predictions]
                for f in targeted_fruits:
                    f["center"] = f["predicted_pos"]
                
                # Check for pomegranate behavior
                is_rapid_mode = self.check_pomegranate_behavior(targeted_fruits, current_time)
                
                # Check game state
                if len(targeted_fruits) == 0 and self.frame_counter % 100 == 0:
                    if is_game_over(frame, GAME_OVER_TEMPLATE_PATH):
                        print("\n🎮 Game Over detected! Pausing...")
                        time.sleep(2)
                        continue
                
                # Find the single best target
                best_target, best_score = self.find_best_swipe_target(targeted_fruits, bombs, current_time)
                
                # Execute swipe
                fruits_cut = 0
                if best_target and best_score > 0.5:
                    fruits_cut = self.execute_swipe(best_target, bombs, current_time, is_rapid_mode)
                    self.total_fruits_cut += fruits_cut
                    action = f"{'RAPID' if is_rapid_mode else 'PRECISION'}"
                else:
                    action = "WAIT"
                
                # Performance monitoring
                frame_time = time.time() - loop_start
                frame_times.append(frame_time)
                if len(frame_times) > 30:
                    frame_times.pop(0)
                
                fps = 1 / (sum(frame_times) / len(frame_times)) if frame_times else 0
                
                # Display info
                fruit_count = len(targeted_fruits)
                bomb_count = len(bombs)
                self.print_timing_info(fps, fruit_count, bomb_count, action)
                
                self.frame_counter += 1
                    
        except pyautogui.FailSafeException:
            print("\n✓ Failsafe triggered! Bot stopped by user.")
        except KeyboardInterrupt:
            print("\n✓ Bot stopped by keyboard interrupt.")
        except Exception as e:
            print(f"\n✓ Error: {e}")
        finally:
            self.game_capture.release()
            print(f"\n🏆 Bot shutdown. Total swipes: {self.total_swipes}, Fruits cut: {self.total_fruits_cut}")

if __name__ == "__main__":
    bot = FruitNinjaBot()
    bot.run()