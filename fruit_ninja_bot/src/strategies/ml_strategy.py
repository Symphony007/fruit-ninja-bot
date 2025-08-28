import time
import numpy as np
from collections import deque

class MLStrategy:
    """
    ENHANCED strategy for precision swiping: 
    - Targets fruits at their PEAK height (slowest point)
    - Better timing for fast-moving fruits
    - Improved accuracy for single fruit swipes
    """
    
    def __init__(self, frame_width, frame_height, min_swipe_interval=0.3):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.min_swipe_interval = min_swipe_interval
        self.last_swipe_time = 0
        self.fruit_tracks = {}  # Track fruits across frames for velocity prediction
        self.track_id_counter = 0
        self.max_track_history = 15  # Increased for better trajectory analysis
    
    def _assign_ids_to_fruits(self, current_fruits, current_time):
        """Track fruits across frames and assign consistent IDs."""
        updated_fruits = []
        
        for fruit in current_fruits:
            fruit_pos = np.array(fruit["center"])
            matched_id = None
            min_distance = float('inf')
            
            # Find closest existing track
            for track_id, track_data in self.fruit_tracks.items():
                if current_time - track_data["last_seen"] < 0.8:  # Longer tracking window
                    last_pos = track_data["positions"][-1]
                    distance = np.linalg.norm(fruit_pos - last_pos)
                    
                    if distance < 80 and distance < min_distance:  # Increased max movement for fast fruits
                        min_distance = distance
                        matched_id = track_id
            
            if matched_id is not None:
                # Update existing track
                track = self.fruit_tracks[matched_id]
                track["positions"].append(fruit_pos)
                if len(track["positions"]) > self.max_track_history:
                    track["positions"].popleft()
                track["last_seen"] = current_time
                track["current_data"] = fruit
                fruit["track_id"] = matched_id
            else:
                # Create new track
                self.track_id_counter += 1
                self.fruit_tracks[self.track_id_counter] = {
                    "positions": deque([fruit_pos], maxlen=self.max_track_history),
                    "last_seen": current_time,
                    "current_data": fruit,
                    "first_seen": current_time  # Track when fruit first appeared
                }
                fruit["track_id"] = self.track_id_counter
            
            updated_fruits.append(fruit)
        
        # Clean up old tracks
        to_remove = [tid for tid, data in self.fruit_tracks.items() 
                    if current_time - data["last_seen"] > 1.5]  # Longer cleanup time
        for tid in to_remove:
            del self.fruit_tracks[tid]
        
        return updated_fruits
    
    def _calculate_velocity(self, track):
        """Calculate velocity from track history."""
        positions = list(track["positions"])
        if len(positions) < 2:
            return (0, 0)
        
        # Use multiple points for smoother velocity calculation
        if len(positions) >= 4:
            recent_positions = positions[-4:]
        else:
            recent_positions = positions
        
        dx = recent_positions[-1][0] - recent_positions[0][0]
        dy = recent_positions[-1][1] - recent_positions[0][1]
        dt = (len(recent_positions) - 1) * (1/60)  # Assuming ~30 FPS
        
        if dt > 0:
            return (dx/dt, dy/dt)
        return (0, 0)
    
    def _is_at_peak(self, track, current_time):
        """
        CRITICAL: Determine if fruit is at peak height (slowest vertical movement).
        Fruits are easiest to hit when they slow down at the top of their arc.
        """
        if len(track["positions"]) < 3:
            return False
        
        vx, vy = self._calculate_velocity(track)
        current_y = track["positions"][-1][1]
        
        # Fruit is at peak when:
        # 1. Vertical velocity is near zero (changing direction)
        # 2. Has been on screen for a bit (not just appearing)
        # 3. Is in the upper half of the screen (not too low)
        is_slow = abs(vy) < 50  # Very slow vertical movement
        is_mature = (current_time - track["first_seen"]) > 0.4  # Been visible for a while
        is_high_enough = current_y < self.frame_height * 0.6  # In upper 60% of screen
        
        return is_slow and is_mature and is_high_enough
    
    def find_best_swipe(self, fruits, current_time):
        """
        Find the optimal swipe - PRIORITIZE FRUITS AT THEIR PEAK.
        """
        if not fruits:
            return None
        
        # Track fruits and predict movement
        tracked_fruits = self._assign_ids_to_fruits(fruits, current_time)
        
        # STRATEGY 1: First, look for fruits at their PEAK (easiest to hit)
        peak_fruits = []
        for fruit in tracked_fruits:
            track_id = fruit.get("track_id")
            if track_id in self.fruit_tracks:
                track = self.fruit_tracks[track_id]
                if self._is_at_peak(track, current_time):
                    peak_fruits.append(fruit)
        
        # If we have fruits at peak, target them
        if peak_fruits:
            # Target the lowest peak fruit (most urgent)
            target_fruit = max(peak_fruits, key=lambda f: f["center"][1])
            cx, cy = target_fruit["center"]
            swipe_length = 150
            return ([target_fruit], cx - swipe_length, cy, cx + swipe_length, cy)
        
        # STRATEGY 2: If no peak fruits, look for multi-fruit opportunities
        multi_swipe = self._find_multi_fruit_swipe(tracked_fruits)
        if multi_swipe:
            return multi_swipe
        
        # STRATEGY 3: Emergency swipe for fruits about to leave screen
        urgent_fruits = [f for f in tracked_fruits if f["center"][1] > self.frame_height * 0.7]
        if urgent_fruits:
            target_fruit = max(urgent_fruits, key=lambda f: f["center"][1])
            cx, cy = target_fruit["center"]
            swipe_length = 150
            return ([target_fruit], cx - swipe_length, cy, cx + swipe_length, cy)
        
        return None
    
    def _find_multi_fruit_swipe(self, fruits):
        """Find swipes that can hit multiple fruits."""
        if len(fruits) < 2:
            return None
        
        # Group fruits by similar height
        height_groups = {}
        for fruit in fruits:
            height_bucket = fruit["center"][1] // 40  # Group every 40px height
            if height_bucket not in height_groups:
                height_groups[height_bucket] = []
            height_groups[height_bucket].append(fruit)
        
        # Find the group with most fruits at similar height
        best_group = None
        for height, group in height_groups.items():
            if len(group) >= 2 and (best_group is None or len(group) > len(best_group)):
                best_group = group
        
        if best_group:
            x_positions = [f["center"][0] for f in best_group]
            min_x, max_x = min(x_positions), max(x_positions)
            avg_y = sum(f["center"][1] for f in best_group) / len(best_group)
            
            swipe_margin = 50
            start_x = max(0, min_x - swipe_margin)
            end_x = min(self.frame_width, max_x + swipe_margin)
            
            return (best_group, start_x, avg_y, end_x, avg_y)
        
        return None
    
    def should_swipe(self, fruits, current_time):
        """Check if we should swipe now."""
        if current_time - self.last_swipe_time < self.min_swipe_interval:
            return False, None
        
        best_swipe = self.find_best_swipe(fruits, current_time)
        if best_swipe:
            return True, best_swipe
        
        return False, None
    
    def update_swipe_time(self, current_time):
        self.last_swipe_time = current_time
        