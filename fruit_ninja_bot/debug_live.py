import cv2
import time
from src.vision.screen_capture import GameCapture
from src.vision.object_detection import find_targets, show_mask_debug
from src.vision.ml_detector import MLDetector
from src.utils.config import *

def main():
    print("=== FRUIT NINJA BOT - LIVE DEBUG ===")
    print("Controls:")
    print("  'q' - Quit")
    print("  'm' - Toggle between ML and CV detection")
    print("  'd' - Toggle debug mask view")
    print("SAFETY: Move mouse to top-left corner to stop.")
    print("\nStarting in 2 seconds...")
    time.sleep(2)
    
    # Initialize components
    capture = GameCapture(CAPTURE_REGION, target_fps=30)
    ml_detector = MLDetector(ML_CONFIDENCE_THRESHOLD)
    
    # Try to initialize ML
    try:
        ml_detector.initialize()
        print(f"✅ ML Detector initialized on {ml_detector.device}")
    except Exception as e:
        print(f"⚠️  ML Detector failed: {e}. Falling back to CV.")
        ml_detector.initialized = False
    
    use_ml = USE_ML_DETECTION and ml_detector.initialized
    show_mask = False
    print(f"Initial detection mode: {'ML' if use_ml else 'CV'}")

    # Performance tracking variables
    frame_count = 0
    start_time = time.time()
    capture_times = []
    detection_times = []

    try:
        while True:
            loop_start = time.time()
            
            # Capture frame
            capture_start = time.time()
            frame = capture.get_frame()
            capture_time = time.time() - capture_start
            if frame is None:
                continue
                
            # Create a copy for display
            display_frame = frame.copy()
            detections = []
            detection_summary = "No detections"
            
            # Perform detection based on current mode
            detect_start = time.time()
            if use_ml:
                detections = ml_detector.detect_objects(frame)
                display_frame = ml_detector.draw_detections(display_frame, detections)
                fruit_count = len([d for d in detections if d["type"] == "fruit"])
                bomb_count = len([d for d in detections if d["type"] == "bomb"])
                detection_summary = f"ML: {fruit_count}F {bomb_count}B"
            else:
                # Use CV detection
                fruits = find_targets(frame, LOWER_RED, UPPER_RED, MIN_FRUIT_AREA)
                fruits2 = find_targets(frame, LOWER_RED2, UPPER_RED2, MIN_FRUIT_AREA)
                fruits.extend(fruits2)
                bombs = find_targets(frame, LOWER_BOMB, UPPER_BOMB, MIN_BOMB_AREA)
                
                # Draw CV detections
                for fruit in fruits:
                    cX, cY = fruit["center"]
                    cv2.circle(display_frame, (cX, cY), 7, (0, 255, 0), -1)
                    cv2.putText(display_frame, "Fruit", (cX - 20, cY - 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                for bomb in bombs:
                    cX, cY = bomb["center"]
                    cv2.circle(display_frame, (cX, cY), 7, (0, 0, 255), -1)
                    cv2.putText(display_frame, "Bomb", (cX - 20, cY - 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                detections = fruits + bombs
                fruit_count = len(fruits)
                bomb_count = len(bombs)
                detection_summary = f"CV: {fruit_count}F {bomb_count}B"
            detect_time = time.time() - detect_start

            # Store performance metrics
            capture_times.append(capture_time)
            detection_times.append(detect_time)
            # Keep only the last 30 readings for a rolling average
            if len(capture_times) > 30:
                capture_times.pop(0)
                detection_times.pop(0)
            
            frame_count += 1
            
            # Show mask debug window if enabled
            if show_mask:
                show_mask_debug(frame, LOWER_RED, UPPER_RED, "Fruit Mask")
                # Can add more mask windows here for other colors
            
            # Calculate current FPS and averages
            current_fps = 1 / (time.time() - loop_start) if (time.time() - loop_start) > 0 else 0
            avg_capture = sum(capture_times) / len(capture_times) * 1000 if capture_times else 0
            avg_detect = sum(detection_times) / len(detection_times) * 1000 if detection_times else 0
            total_time = avg_capture + avg_detect

            # Add status text to the main display
            mode_text = "ML" if use_ml else "CV"
            status_text = f"Mode: {mode_text} | {detection_summary}"
            fps_text = f"FPS: {current_fps:.1f} | Cap: {avg_capture:.1f}ms | Det: {avg_detect:.1f}ms | Tot: {total_time:.1f}ms"
            
            cv2.putText(display_frame, status_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_frame, fps_text, (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show the frame
            cv2.imshow('Fruit Ninja Bot - LIVE DEBUG', display_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('m'):
                if ml_detector.initialized:
                    use_ml = not use_ml
                    print(f"Toggled detection mode to: {'ML' if use_ml else 'CV'}")
                else:
                    print("ML not available. Cannot toggle.")
            elif key == ord('d'):
                show_mask = not show_mask
                if not show_mask:
                    cv2.destroyWindow('Fruit Mask')  # Close debug window
                print(f"Debug mask: {'ON' if show_mask else 'OFF'}")
                
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        capture.release()
        cv2.destroyAllWindows()
        total_duration = time.time() - start_time
        avg_fps = frame_count / total_duration if total_duration > 0 else 0
        print(f"\n=== PERFORMANCE SUMMARY ===")
        print(f"Total frames: {frame_count}")
        print(f"Total time: {total_duration:.2f}s")
        print(f"Average FPS: {avg_fps:.2f}")
        if capture_times:
            print(f"Avg Capture Time: {sum(capture_times)/len(capture_times)*1000:.2f}ms")
        if detection_times:
            print(f"Avg Detection Time: {sum(detection_times)/len(detection_times)*1000:.2f}ms")
        print("Debug session ended.")

if __name__ == "__main__":
    main()