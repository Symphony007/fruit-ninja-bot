import time
try:
    import mouse
    MOUSE_AVAILABLE = True
except ImportError:
    MOUSE_AVAILABLE = False
    import pyautogui

class MouseController:
    """
    Optimized mouse controller for game swipes with reliable registration.
    """
    
    def __init__(self):
        self.mouse_available = MOUSE_AVAILABLE
        if self.mouse_available:
            print("✓ Using optimized mouse control")
        else:
            print("⚠️ Using fallback pyautogui control")
    
    def perform_swipe(self, start_x, start_y, end_x, end_y, duration=0.05):
        """
        Perform a reliable swipe motion that the game can register.
        """
        if self.mouse_available:
            # Optimized method using mouse library
            self._optimized_swipe(start_x, start_y, end_x, end_y, duration)
        else:
            # Fallback method
            self._fallback_swipe(start_x, start_y, end_x, end_y, duration)
    
    def _optimized_swipe(self, start_x, start_y, end_x, end_y, duration):
        """Optimized swipe that games can register reliably"""
        try:
            # Move to start position
            mouse.move(start_x, start_y, absolute=True, duration=0.01)
            time.sleep(0.02)  # Small delay for stability
            
            # Press and hold mouse
            mouse.press(button='left')
            time.sleep(0.03)  # Ensure press is registered
            
            # Move to end position with optimal speed
            mouse.move(end_x, end_y, absolute=True, duration=duration)
            time.sleep(0.02)  # Ensure movement is registered
            
            # Release mouse
            mouse.release(button='left')
            
        except Exception as e:
            print(f"Optimized swipe error: {e}")
            self._fallback_swipe(start_x, start_y, end_x, end_y, duration)
    
    def _fallback_swipe(self, start_x, start_y, end_x, end_y, duration):
        """Fallback swipe method using pyautogui"""
        import pyautogui
        try:
            pyautogui.moveTo(start_x, start_y, duration=0.02)
            pyautogui.mouseDown()
            pyautogui.moveTo(end_x, end_y, duration=duration)
            pyautogui.mouseUp()
        except Exception as e:
            print(f"Fallback swipe error: {e}")
    
    def safe_swipe(self, target_x, target_y, swipe_length=180, duration=0.05):
        """Safe wrapper for swipe with coordinates validation"""
        try:
            # Calculate swipe coordinates
            start_x = target_x - swipe_length
            end_x = target_x + swipe_length
            
            # Ensure coordinates are within screen bounds
            start_x = max(0, min(start_x, 1920))
            end_x = max(0, min(end_x, 1920))
            
            self.perform_swipe(start_x, target_y, end_x, target_y, duration)
            return True
            
        except Exception as e:
            print(f"Swipe failed: {e}")
            return False