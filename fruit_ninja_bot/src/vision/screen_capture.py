import cv2
import numpy as np
import time
from mss import mss

class GameCapture:
    """
    Simple and reliable screen capture using MSS without threading.
    """
    
    def __init__(self, capture_region=None, target_fps=60):
        self.capture_region = capture_region
        self.target_fps = target_fps
        self.frame_time = 1.0 / target_fps
        self.last_capture_time = 0
        
        # Initialize MSS
        try:
            self.camera = mss()
            if capture_region:
                self.monitor = {
                    "left": capture_region[0],
                    "top": capture_region[1],
                    "width": capture_region[2] - capture_region[0],
                    "height": capture_region[3] - capture_region[1]
                }
            else:
                self.monitor = self.camera.monitors[1]
            print("✓ MSS capture initialized")
            
        except Exception as e:
            print(f"MSS failed: {e}")
            raise Exception("Screen capture initialization failed")
    
    def get_frame(self):
        """Capture a frame with timing control."""
        current_time = time.time()
        
        # Enforce frame rate limit
        if current_time - self.last_capture_time < self.frame_time:
            return None
            
        try:
            # Capture frame
            sct_img = self.camera.grab(self.monitor)
            if sct_img:
                # ORIGINAL WORKING METHOD - correct color conversion
                frame = np.array(sct_img, dtype=np.uint8)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                self.last_capture_time = current_time
                return frame
                
        except Exception as e:
            print(f"Capture error: {e}")
            
        return None
    
    def release(self):
        """Clean up resources."""
        try:
            self.camera.close()
        except:
            pass

# Test function
if __name__ == "__main__":
    from src.utils.config import CAPTURE_REGION
    
    print("Testing MSS capture speed...")
    capture = GameCapture(CAPTURE_REGION, target_fps=60)
    
    try:
        frames = 0
        start_time = time.time()
        test_duration = 5.0
        
        while time.time() - start_time < test_duration:
            frame = capture.get_frame()
            if frame is not None:
                frames += 1
                if frames % 10 == 0:
                    current_fps = frames / (time.time() - start_time)
                    print(f"Frames: {frames}, Current FPS: {current_fps:.1f}")
        
        end_time = time.time()
        total_time = end_time - start_time
        fps = frames / total_time if total_time > 0 else 0
        print(f"✅ Final: {frames} frames in {total_time:.2f}s = {fps:.1f} FPS")
        
    except Exception as e:
        print(f"Test error: {e}")
    finally:
        capture.release()