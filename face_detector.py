"""
Face Detection Module for Active Speaker Detection
Uses MediaPipe for fast and accurate face detection
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import List, Tuple, Optional


class FaceDetector:
    """Detect and track faces in video frames"""
    
    def __init__(self, min_detection_confidence=0.5):
        """
        Initialize MediaPipe Face Detection
        
        Args:
            min_detection_confidence: Minimum confidence value [0.0, 1.0] for face detection
        """
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            min_detection_confidence=min_detection_confidence,
            model_selection=0  # 0 for short-range (within 2 meters), 1 for full-range
        )
        
    def detect_faces(self, frame: np.ndarray) -> List[dict]:
        """
        Detect all faces in a single frame
        
        Args:
            frame: BGR image from OpenCV
            
        Returns:
            List of face dictionaries with keys:
                - 'bbox': (x, y, width, height) in pixels
                - 'confidence': detection confidence [0.0, 1.0]
                - 'center': (cx, cy) center point of face
        """
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        results = self.face_detection.process(rgb_frame)
        
        if not results.detections:
            return []
        
        h, w, _ = frame.shape
        faces = []
        
        for detection in results.detections:
            # Get bounding box
            bbox = detection.location_data.relative_bounding_box
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)
            
            # Ensure bbox is within frame bounds
            x = max(0, x)
            y = max(0, y)
            width = min(width, w - x)
            height = min(height, h - y)
            
            # EXPAND BOUNDING BOX UPWARD to include full head (hair, forehead, etc.)
            # MediaPipe detects face only, we need to expand for head
            expansion_top = int(height * 0.4)  # Add 40% of face height above
            expansion_sides = int(width * 0.1)  # Add 10% on each side for margin
            
            # Calculate expanded bbox
            expanded_y = max(0, y - expansion_top)
            expanded_x = max(0, x - expansion_sides)
            expanded_height = min(height + expansion_top, h - expanded_y)
            expanded_width = min(width + 2 * expansion_sides, w - expanded_x)
            
            # Center point of EXPANDED bbox (better for cropping)
            center_x = expanded_x + expanded_width // 2
            center_y = expanded_y + expanded_height // 2
            
            faces.append({
                'bbox': (x, y, width, height),  # Original face bbox
                'expanded_bbox': (expanded_x, expanded_y, expanded_width, expanded_height),  # Expanded for head
                'confidence': detection.score[0],
                'center': (center_x, center_y)  # Center of expanded bbox
            })
        
        return faces
    
    def get_largest_face(self, faces: List[dict]) -> Optional[dict]:
        """
        Get the largest face (usually the main speaker)
        
        Args:
            faces: List of face dictionaries from detect_faces()
            
        Returns:
            Face dictionary with largest area, or None if no faces
        """
        if not faces:
            return None
        
        return max(faces, key=lambda f: f['bbox'][2] * f['bbox'][3])
    
    def __del__(self):
        """Clean up MediaPipe resources"""
        if hasattr(self, 'face_detection'):
            self.face_detection.close()


def analyze_video_faces(video_path: str, sample_rate: int = 30) -> List[Tuple[float, List[dict]]]:
    """
    Analyze faces throughout a video by sampling frames
    
    Args:
        video_path: Path to video file
        sample_rate: Analyze every Nth frame (default 30 = ~1 sample per second for 30fps)
        
    Returns:
        List of (timestamp, faces) tuples where:
            - timestamp: time in seconds
            - faces: list of face dicts from detect_faces()
    """
    detector = FaceDetector()
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30  # fallback
    
    results = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % sample_rate == 0:
            timestamp = frame_count / fps
            faces = detector.detect_faces(frame)
            results.append((timestamp, faces))
        
        frame_count += 1
    
    cap.release()
    return results


if __name__ == "__main__":
    # Test with a video file
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python face_detector.py <video_file>")
        sys.exit(1)
    
    video_file = sys.argv[1]
    print(f"Analyzing faces in: {video_file}")
    
    results = analyze_video_faces(video_file, sample_rate=30)
    
    print(f"\nFound faces in {len(results)} sampled frames:")
    for timestamp, faces in results[:10]:  # Show first 10
        print(f"  {timestamp:.2f}s: {len(faces)} face(s)")
        for i, face in enumerate(faces):
            print(f"    Face {i+1}: bbox={face['bbox']}, confidence={face['confidence']:.2f}")
