"""
Active Speaker Detection Module
Detects who is speaking by analyzing audio energy and correlating with face positions
"""

import cv2
import numpy as np
import subprocess
import json
import os
from typing import List, Tuple, Optional, Dict
from face_detector import FaceDetector, PersonDetector


class ActiveSpeakerDetector:
    """Detect active speaker by correlating audio with face/person positions"""
    
    def __init__(self, min_audio_threshold=0.01):
        """
        Initialize active speaker detector
        
        Args:
            min_audio_threshold: Minimum audio energy to consider as speech
        """
        self.face_detector = FaceDetector(min_detection_confidence=0.5)
        self.person_detector = PersonDetector(min_detection_confidence=0.5)
        self.min_audio_threshold = min_audio_threshold
    
    def extract_audio_energy(self, video_path: str, segment_duration: float = 0.5) -> List[Tuple[float, float]]:
        """
        Extract audio energy levels over time
        
        Args:
            video_path: Path to video file
            segment_duration: Duration of each audio segment in seconds
            
        Returns:
            List of (timestamp, energy) tuples
        """
        # Use ffmpeg to extract audio stats
        cmd = [
            'ffprobe',
            '-f', 'lavfi',
            '-i', f'amovie={video_path},astats=metadata=1:reset=1',
            '-show_entries', 'frame=pkt_pts_time:frame_tags=lavfi.astats.Overall.RMS_level',
            '-of', 'json',
            '-v', 'quiet'
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)
            
            energy_data = []
            for frame in data.get('frames', []):
                timestamp = float(frame.get('pkt_pts_time', 0))
                rms = frame.get('tags', {}).get('lavfi.astats.Overall.RMS_level', '-inf')
                
                # Convert dB to linear scale
                if rms != '-inf':
                    energy = 10 ** (float(rms) / 20)
                else:
                    energy = 0
                
                energy_data.append((timestamp, energy))
            
            return energy_data
        except Exception as e:
            print(f"Warning: Could not extract audio energy: {e}")
            # Fallback: return empty list
            return []
    
    def detect_active_speaker(
        self, 
        video_path: str, 
        sample_rate: int = 15
    ) -> List[Tuple[float, Optional[dict]]]:
        """
        Detect active speaker throughout video
        
        Args:
            video_path: Path to video file
            sample_rate: Analyze every Nth frame (default 15 = ~0.5s for 30fps)
            
        Returns:
            List of (timestamp, active_face) tuples where:
                - timestamp: time in seconds
                - active_face: face dict from FaceDetector, or None if no speaker
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 30
        
        # Extract audio energy
        print("  Analyzing audio energy...")
        audio_energy = self.extract_audio_energy(video_path)
        
        # Build audio energy lookup (timestamp -> energy)
        audio_map = {t: e for t, e in audio_energy}
        
        results = []
        frame_count = 0
        
        print("  Detecting faces and correlating with audio...")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % sample_rate == 0:
                timestamp = frame_count / fps
                
                # Detect faces
                faces = self.face_detector.detect_faces(frame)
                
                # Get audio energy at this timestamp
                audio_energy_val = audio_map.get(round(timestamp, 1), 0)
                
                # Determine active speaker
                if audio_energy_val > self.min_audio_threshold and faces:
                    # If audio is active, pick the largest/most central face as speaker
                    active_face = self.face_detector.get_largest_face(faces)
                else:
                    # No speech or no faces
                    active_face = None
                
                results.append((timestamp, active_face))
            
            frame_count += 1
        
        cap.release()
        return results
    
    def get_primary_speaker_position(
        self, 
        video_path: str, 
        sample_frames: int = 60
    ) -> Optional[Tuple[int, int]]:
        """
        Quick analysis: Get the most common face position (simplified approach)
        Useful for videos where speaker doesn't move much
        
        Args:
            video_path: Path to video file
            sample_frames: Number of frames to sample throughout video (default 60)
            
        Returns:
            (x, y) center position of primary speaker, or None if unstable/no face
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return None
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            cap.release()
            return None
        
        # Sample frames evenly throughout video
        frame_indices = np.linspace(0, total_frames - 1, min(sample_frames, total_frames), dtype=int)
        
        face_positions = []
        person_positions = []
        detection_method = None  # Track which method worked
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            # Try face detection first
            faces = self.face_detector.detect_faces(frame)
            if faces:
                largest_face = self.face_detector.get_largest_face(faces)
                face_positions.append(largest_face['center'])
            else:
                # Fallback to person detection
                person = self.person_detector.detect_person(frame)
                if person:
                    person_positions.append(person['center'])
        
        cap.release()
        
        # Determine which detection method to use
        if len(face_positions) >= len(person_positions):
            positions = face_positions
            detection_method = "face"
        else:
            positions = person_positions
            detection_method = "person"
        
        if not positions:
            print(f"  ⚠️  No face or person detected in any frames")
            return None
        
        print(f"  Using {detection_method} detection ({len(positions)}/{len(frame_indices)} frames)")
        
        # Check if face detection is stable (low variance)
        # If speaker moves a lot, variance will be high -> fallback to center crop
        positions_array = np.array(positions)
        
        # Calculate variance
        var_x = np.var(positions_array[:, 0])
        var_y = np.var(positions_array[:, 1])
        
        # Get video dimensions for relative variance
        sample_frame_idx = frame_indices[0]
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, sample_frame_idx)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            frame_h, frame_w = frame.shape[:2]
            # Calculate relative variance (as % of frame size)
            rel_var_x = var_x / (frame_w ** 2) * 100
            rel_var_y = var_y / (frame_h ** 2) * 100
            
            # If variance is too high (speaker moves around a lot), return None
            # Threshold: 2% variance means face position varies by ~14% of frame width
            if rel_var_x > 2.0 or rel_var_y > 2.0:
                print(f"  ⚠️  High variance detected (x: {rel_var_x:.2f}%, y: {rel_var_y:.2f}%)")
                print(f"  Speaker position is unstable, using center crop instead")
                return None
        
        # Check detection consistency - need at least 60% of frames with face
        detection_rate = len(positions) / len(frame_indices)
        if detection_rate < 0.6:
            print(f"  ⚠️  Low detection rate: {detection_rate*100:.0f}%")
            print(f"  Face not consistently detected, using center crop instead")
            return None
        
        # Return median position (more robust than mean)
        median_x = int(np.median(positions_array[:, 0]))
        median_y = int(np.median(positions_array[:, 1]))
        
        print(f"  ✓ Stable face position detected ({detection_rate*100:.0f}% detection rate)")
        
        return (median_x, median_y)


if __name__ == "__main__":
    # Test active speaker detection
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python speaker_detector.py <video_file>")
        sys.exit(1)
    
    video_file = sys.argv[1]
    print(f"Analyzing active speaker in: {video_file}\n")
    
    detector = ActiveSpeakerDetector()
    
    # Quick test: Get primary speaker position
    print("[Quick Analysis]")
    primary_pos = detector.get_primary_speaker_position(video_file)
    if primary_pos:
        print(f"Primary speaker position: {primary_pos}")
    else:
        print("No speaker detected")
    
    # Full test: Detect active speaker over time
    print("\n[Full Analysis]")
    results = detector.detect_active_speaker(video_file)
    
    print(f"\nAnalyzed {len(results)} segments:")
    speaker_count = sum(1 for _, face in results if face is not None)
    print(f"  Segments with active speaker: {speaker_count}/{len(results)}")
