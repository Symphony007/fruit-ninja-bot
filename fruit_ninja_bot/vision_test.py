import cv2
import time
from src.vision.screen_capture import GameCapture
from src.vision.object_detection import find_targets
from src.vision.ml_detector import MLDetector
from src.utils.config import *

def main():
    print("Starting Vision Test...")
    print("Press 'q' to quit. Press 'm' to toggle ML/CV detection.")
    
    # Initialize components
    capture = GameCapture(CAPTURE_REGION, target_fps=30)
    ml_detector = MLDetector(ML_CONFIDENCE_THRESHOLD)
    ml_detector.initialize()
    
    use_ml = USE_ML_DETECTION
    print(f"Initial detection mode: {'ML' if use_ml else 'CV'}")
    
    try:
        while True:
            frame = capture.get_frame()
            if frame is None:
                continue
                
            display_frame = frame.copy()
            detections = []
            
            # Perform detection based on current mode
            if use_ml and ml_detector.initialized:
                detections = ml_detector.detect_objects(frame)
                display_frame = ml_detector.draw_detections(display_frame, detections)
            else:
                # Use CV detection
                fruits = find_targets(frame, LOWER_RED, UPPER_RED, MIN_FRUIT_AREA)
                fruits2 = find_targets(frame, LOWER_RED2, UPPER_RED2, MIN_FRUIT_AREA)
                fruits.extend(fruits2)
                bombs = find_targets(frame, LOWER_BOMB, UPPER_BOMB, MIN_BOMB_AREA)
                
                # Draw CV detections
                for fruit in fruits:
                    cX, cY = fruit["center"]
                    cv2.circle(display_frame, (cX, cY), 7, (0, 255, 0), -1) # Green for fruit
                    cv2.putText(display_frame, "Fruit", (cX - 20, cY - 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                for bomb in bombs:
                    cX, cY = bomb["center"]
                    cv2.circle(display_frame, (cX, cY), 7, (0, 0, 255), -1) # Red for bomb
                    cv2.putText(display_frame, "Bomb", (cX - 20, cY - 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                detections = fruits + bombs
            
            # Print detection count to console
            fruit_count = len([d for d in detections if d.get('type') == 'fruit'])
            bomb_count = len([d for d in detections if d.get('type') == 'bomb'])
            print(f"Fruits: {fruit_count}, Bombs: {bomb_count} - Mode: {'ML' if use_ml else 'CV'}", end='\r')
            
            # Create a smaller resized frame for the preview window
            scale_percent = 50  # Adjust this percentage to make window smaller/larger
            width = int(display_frame.shape[1] * scale_percent / 100)
            height = int(display_frame.shape[0] * scale_percent / 100)
            dim = (width, height)
            resized_frame = cv2.resize(display_frame, dim, interpolation=cv2.INTER_AREA)
            
            # Show the resized frame in a window positioned at top-left corner
            cv2.imshow('Fruit Ninja Bot - Vision Test', resized_frame)
            cv2.moveWindow('Fruit Ninja Bot - Vision Test', 0, 0)  # Position at (0, 0)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('m'):
                use_ml = not use_ml
                print(f"\nToggled detection mode to: {'ML' if use_ml else 'CV'}")
                
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        capture.release()
        cv2.destroyAllWindows()
        print("Vision test completed.")

if __name__ == "__main__":
    main()
