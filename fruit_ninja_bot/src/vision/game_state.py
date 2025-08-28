import cv2
import numpy as np

def is_game_over(frame, game_over_template_path):
    """
    Check if the game over screen is displayed by template matching.
    
    Args:
        frame: The current game frame
        game_over_template_path: Path to the game over template image
        
    Returns:
        bool: True if game over screen is detected, False otherwise
    """
    try:
        # Load the game over template
        template = cv2.imread(game_over_template_path)
        if template is None:
            print("Game over template not found")
            return False
            
        # Perform template matching
        result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        
        # If match confidence is high enough, game is over
        if max_val > 0.8:  # Adjust this threshold as needed
            return True
            
    except Exception as e:
        print(f"Game state check error: {e}")
        
    return False