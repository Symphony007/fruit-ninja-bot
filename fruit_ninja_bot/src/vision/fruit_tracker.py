import numpy as np
import time
from collections import deque
import math
import cv2

class FruitTracker:
    """
    Advanced fruit trajectory prediction with physics modeling.
    Tracks fruits across frames and predicts future positions.
    """
    
    def __init__(self, max_history=10):
        self.fruit_history = {}  # {track_id: deque of (time, x, y)}
        self.next_track_id = 0
        self.max_history = max_history
        self.frame_time = 1/30  # Assuming 30 FPS
        
        # Physics constants (adjust based on game observation)
        self.gravity = 9.8 * 50  # Scaled for pixels (empirical)
        self.initial_upward_velocity = 400  # pixels/second (empirical)
        
    def _calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def _assign_track_ids(self, current_fruits):
        """
        Match current fruits with existing tracks using Hungarian algorithm.
        Returns dictionary: {track_id: current_fruit_data}
        """
        if not self.fruit_history:
            # First frame, assign new IDs to all fruits
            assigned = {}
            for fruit in current_fruits:
                assigned[self.next_track_id] = fruit
                self.next_track_id += 1
            return assigned
        
        # Create cost matrix for simple assignment
        existing_ids = list(self.fruit_history.keys())
        cost_matrix = np.zeros((len(existing_ids), len(current_fruits)))
        
        for i, track_id in enumerate(existing_ids):
            if self.fruit_history[track_id]:
                last_pos = self.fruit_history[track_id][-1][1:]  # (x, y) of last position
                for j, fruit in enumerate(current_fruits):
                    current_pos = fruit["center"]
                    cost_matrix[i, j] = self._calculate_distance(last_pos, current_pos)
        
        # Simple assignment: closest match within threshold
        assigned = {}
        used_current = set()
        
        for i, track_id in enumerate(existing_ids):
            if not current_fruits:
                break
                
            # Find closest current fruit to this track
            if len(current_fruits) > 0:
                closest_idx = np.argmin(cost_matrix[i])
                closest_distance = cost_matrix[i, closest_idx]
                
                # If close enough, assign to track
                if closest_distance < 50:  # 50 pixel threshold
                    assigned[track_id] = current_fruits[closest_idx]
                    used_current.add(closest_idx)
        
        # Assign new IDs to unmatched fruits
        for j, fruit in enumerate(current_fruits):
            if j not in used_current:
                new_id = self.next_track_id
                assigned[new_id] = fruit
                self.next_track_id += 1
        
        return assigned
    
    def _calculate_trajectory_parameters(self, track_id):
        """
        Calculate physics parameters for a tracked fruit.
        Returns: (current_phase, velocity_x, velocity_y, acceleration_y)
        """
        history = list(self.fruit_history[track_id])
        if len(history) < 2:
            return "unknown", 0, 0, 0
        
        # Extract recent positions and times
        times = [h[0] for h in history]
        xs = [h[1] for h in history]
        ys = [h[2] for h in history]
        
        # Use last 3 points if available, else use all
        n_points = min(3, len(history))
        recent_times = times[-n_points:]
        recent_xs = xs[-n_points:]
        recent_ys = ys[-n_points:]
        
        # Calculate velocities from most recent movement
        dt = recent_times[-1] - recent_times[-2]
        if dt <= 0:
            return "unknown", 0, 0, 0
            
        velocity_x = (recent_xs[-1] - recent_xs[-2]) / dt
        velocity_y = (recent_ys[-1] - recent_ys[-2]) / dt
        
        # Calculate acceleration (if we have enough points)
        if len(recent_times) >= 3:
            prev_dt = recent_times[-2] - recent_times[-3]
            if prev_dt > 0:
                prev_velocity_y = (recent_ys[-2] - recent_ys[-3]) / prev_dt
                acceleration_y = (velocity_y - prev_velocity_y) / dt
            else:
                acceleration_y = -self.gravity
        else:
            acceleration_y = -self.gravity  # Assume gravity
        
        # Determine phase based on velocity and acceleration
        if velocity_y < -5 and acceleration_y < 0:  # Moving up, accelerating down
            return "ascending", velocity_x, velocity_y, acceleration_y
        elif abs(velocity_y) < 15:  # Near peak
            return "hover", velocity_x, velocity_y, acceleration_y
        elif velocity_y > 5:  # Moving down
            return "descending", velocity_x, velocity_y, acceleration_y
        else:
            return "unknown", velocity_x, velocity_y, acceleration_y
    
    def update(self, current_fruits):
        """
        Update tracks with current frame's fruits.
        Returns: list of fruits with track_id and predicted future position
        """
        current_time = time.time()
        assigned_fruits = self._assign_track_ids(current_fruits)
        
        updated_fruits = []
        
        for track_id, fruit in assigned_fruits.items():
            # Add to history
            if track_id not in self.fruit_history:
                self.fruit_history[track_id] = deque(maxlen=self.max_history)
            
            self.fruit_history[track_id].append((current_time, fruit["center"][0], fruit["center"][1]))
            
            # Calculate trajectory parameters
            if len(self.fruit_history[track_id]) >= 2:
                phase, vel_x, vel_y, accel_y = self._calculate_trajectory_parameters(track_id)
                
                # Predict position 150ms into future (compensate for latency)
                predict_time = 0.15  # 150ms prediction
                current_x, current_y = fruit["center"]
                
                predicted_x = current_x + vel_x * predict_time
                predicted_y = current_y + vel_y * predict_time + 0.5 * accel_y * predict_time**2
                
                # Add prediction to fruit data
                fruit["track_id"] = track_id
                fruit["phase"] = phase
                fruit["predicted_pos"] = (int(predicted_x), int(predicted_y))
                fruit["velocity"] = (vel_x, vel_y)
            else:
                # Not enough history for prediction
                fruit["track_id"] = track_id
                fruit["phase"] = "unknown"
                fruit["predicted_pos"] = fruit["center"]
                fruit["velocity"] = (0, 0)
            
            updated_fruits.append(fruit)
        
        # Clean up old tracks
        self._clean_old_tracks(current_time)
        
        return updated_fruits
    
    def _clean_old_tracks(self, current_time):
        """Remove tracks that haven't been updated recently"""
        to_remove = []
        for track_id, history in self.fruit_history.items():
            if history and current_time - history[-1][0] > 1.0:  # 1 second timeout
                to_remove.append(track_id)
        
        for track_id in to_remove:
            del self.fruit_history[track_id]
    
    def draw_predictions(self, frame, fruits):
        """Draw predicted positions on frame for visualization"""
        for fruit in fruits:
            if "predicted_pos" in fruit:
                pred_x, pred_y = fruit["predicted_pos"]
                curr_x, curr_y = fruit["center"]
                
                # Draw current position (green)
                cv2.circle(frame, (curr_x, curr_y), 8, (0, 255, 0), 2)
                
                # Draw predicted position (blue)
                cv2.circle(frame, (pred_x, pred_y), 6, (255, 0, 0), 2)
                
                # Draw line from current to predicted
                cv2.line(frame, (curr_x, curr_y), (pred_x, pred_y), (255, 255, 0), 2)
                
                # Draw phase text
                phase = fruit.get("phase", "unknown")
                cv2.putText(frame, f"{phase}", (curr_x - 30, curr_y - 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return frame