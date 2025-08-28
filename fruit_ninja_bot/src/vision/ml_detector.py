import cv2
import numpy as np
import time
import torch
from ultralytics import YOLO
from src.utils.config import MODEL_PATH  # Added import

class MLDetector:
    """
    ML-based object detection using YOLO for fruit and bomb detection with GPU support.
    """
    
    def __init__(self, confidence_threshold=0.6):
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.class_names = []
        self.initialized = False
        self.device = None
        
    def initialize(self):
        """Initialize the YOLO model with GPU support."""
        try:
            # Determine device
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Using device: {self.device}")
            
            # ONLY use custom trained model - NO COCO fallback
            try:
                self.model = YOLO(MODEL_PATH)  # Fixed: Use config path
                print(f"Loaded custom trained YOLO model from: {MODEL_PATH}")
            except Exception as e:
                print(f"Failed to load custom model from {MODEL_PATH}: {e}")
                raise Exception("Custom YOLO model required - no fallback to COCO")
            
            # Move model to appropriate device
            self.model.to(self.device)
            
            # Get class names
            self.class_names = self.model.names
            self.initialized = True
            print(f"YOLO model initialized on {self.device.upper()} with {len(self.class_names)} classes")
            
        except Exception as e:
            print(f"Failed to initialize YOLO model: {e}")
            self.initialized = False
    
    def detect_objects(self, frame):
        """
        Detect fruits and bombs in the frame using YOLO with optimized GPU settings.
        """
        if not self.initialized or self.model is None:
            return []
        
        try:
            # RESIZE THE FRAME to match training dimensions (640x640)
            resized_frame = cv2.resize(frame, (640, 640))
            
            # Run inference with OPTIMIZED GPU settings
            results = self.model(resized_frame, 
                               verbose=False, 
                               conf=self.confidence_threshold,
                               device=self.device,
                               half=False,    # Disable half-precision (more stable)
                               imgsz=640,     # Explicit image size
                               max_det=10,    # Limit detections for speed
                               agnostic_nms=True)  # Faster NMS
            
            detections = []
            
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])
                        class_name = self.class_names[class_id]
                        
                        # Filter for relevant objects
                        if class_name in ['fruit', 'bomb']:
                            # Get bounding box coordinates
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            
                            # Scale coordinates back to original frame size
                            scale_x = frame.shape[1] / 640.0
                            scale_y = frame.shape[0] / 640.0
                            
                            x1 = int(x1 * scale_x)
                            y1 = int(y1 * scale_y)
                            x2 = int(x2 * scale_x)
                            y2 = int(y2 * scale_y)
                            
                            # Calculate center point
                            center_x = int((x1 + x2) / 2)
                            center_y = int((y1 + y2) / 2)
                            
                            # Determine type (fruit or bomb)
                            obj_type = "bomb" if class_name == "bomb" else "fruit"
                            
                            detections.append({
                                "type": obj_type,
                                "center": (center_x, center_y),
                                "confidence": confidence,
                                "bbox": (int(x1), int(y1), int(x2), int(y2)),
                                "class_name": class_name
                            })
            
            return detections
            
        except Exception as e:
            print(f"Detection error: {e}")
            return []
    
    def draw_detections(self, frame, detections):
        """Draw detection results with tracking info"""
        for detection in detections:
            x1, y1, x2, y2 = detection["bbox"]
            center_x, center_y = detection["center"]
            
            # Choose color based on type
            color = (0, 255, 0) if detection["type"] == "fruit" else (0, 0, 255)
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw center point
            cv2.circle(frame, (center_x, center_y), 5, color, -1)
            
            # Draw label with tracking info if available
            label = f"{detection['class_name']} {detection['confidence']:.2f}"
            if 'track_id' in detection:
                label = f"T{detection['track_id']}: {label}"
            if 'phase' in detection:
                label += f" ({detection['phase']})"
            
            cv2.putText(frame, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame