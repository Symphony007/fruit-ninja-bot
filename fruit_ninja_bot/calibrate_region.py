import pyautogui
import time
import keyboard  # We'll use the 'keyboard' library for a clean pause

print("Mouse Coordinate Finder")
print("========================")
print("Instructions:")
print("1. Move your mouse to the TOP-LEFT corner of the play area.")
print("2. Press and hold the 'ALT' key for 1 second to capture the position.")
print("3. You will hear a beep and see the coordinates saved.")
print("4. Repeat for the BOTTOM-RIGHT corner.")
print("5. Press 'ESC' key to exit and calculate the region.\n")

print("Waiting for first point (Top-Left)...")

saved_positions = []

try:
    while True:
        # Check if 'esc' is pressed to exit
        if keyboard.is_pressed('esc'):
            print("\nExiting...")
            break
            
        # Check if 'alt' is pressed to capture a point
        if keyboard.is_pressed('alt'):
            # Get the current position
            x, y = pyautogui.position()
            # Add a small delay and check if the key is still pressed (to avoid accidental triggers)
            time.sleep(0.3)
            if keyboard.is_pressed('alt'):
                saved_positions.append((x, y))
                print(f"Point {len(saved_positions)} saved: X={x}, Y={y}")
                # Beep (optional, might not work on all systems)
                print('\a')  # This is the bell character, might cause a beep
                # Wait for key to be released before continuing
                while keyboard.is_pressed('alt'):
                    time.sleep(0.1)
                # If we have two points, we can break early
                if len(saved_positions) == 2:
                    break
                    
        # Short delay to prevent high CPU usage
        time.sleep(0.05)
        
except KeyboardInterrupt:
    print("\nStopped by user.")

# Calculate and print the region if we have two points
if len(saved_positions) == 2:
    (x1, y1), (x2, y2) = saved_positions
    # Ensure we have top-left and bottom-right
    left = min(x1, x2)
    top = min(y1, y2)
    right = max(x1, x2)
    bottom = max(y1, y2)
    
    print("\n" + "="*50)
    print("✅ CAPTURE_REGION Calculated:")
    print(f"CAPTURE_REGION = ({left}, {top}, {right}, {bottom})")
    print("="*50)
    print("\nCopy the line above and replace the value in 'src/utils/config.py'")
else:
    print(f"\nCould not calculate region. Saved {len(saved_positions)} points.")