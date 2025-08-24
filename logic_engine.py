# logic_engine.py
import numpy as np
from collections import defaultdict
import cv2

class SuspiciousActivityLogic:
    def __init__(self, abandonment_threshold_frames, proximity_threshold_pixels, loitering_threshold_frames):
        self.abandonment_threshold = abandonment_threshold_frames
        self.proximity_threshold = proximity_threshold_pixels
        self.loitering_threshold = loitering_threshold_frames
        
        # State tracking dictionaries
        self.object_states = defaultdict(lambda: {'history': [], 'stationary_frames': 0, 'class_name': None, 'alerted': False})
        self.loitering_objects = defaultdict(lambda: {'frames_in_zone': 0, 'alerted': False})

    def get_centroid(self, bbox):
        x1, y1, x2, y2 = bbox
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))

    def update_states(self, tracked_objects, class_names):
        """
        Updates the state of each tracked object, including position history and stationary frame count.
        """
        current_ids = set()
        for obj in tracked_objects:
            x1, y1, x2, y2, track_id = map(int, obj)
            current_ids.add(track_id)
            
            centroid = self.get_centroid((x1, y1, x2, y2))
            state = self.object_states[track_id]
            
            # Assign class name on first sight
            if state['class_name'] is None:
                state['class_name'] = class_names.get(track_id, 'object')

            # Check for stationarity
            if len(state['history']) > 0:
                prev_centroid = state['history'][-1]
                distance = np.sqrt((centroid[0] - prev_centroid[0])**2 + (centroid[1] - prev_centroid[1])**2)
                if distance < 10:  # Movement threshold in pixels
                    state['stationary_frames'] += 1
                else:
                    state['stationary_frames'] = 0 # Reset if object moves
            
            state['history'].append(centroid)
            if len(state['history']) > 100: # Keep history to a reasonable size
                state['history'].pop(0)

        # Clean up old tracks
        for track_id in list(self.object_states.keys()):
            if track_id not in current_ids:
                del self.object_states[track_id]
                if track_id in self.loitering_objects:
                    del self.loitering_objects[track_id]

    def check_abandoned_objects(self):
        """
        Checks for abandoned objects based on stationarity and proximity to people.
        """
        alerts = []
        persons = [state for state in self.object_states.values() if state['class_name'] == 'person']
        person_centroids = [s['history'][-1] for s in persons if s['history']]

        potential_abandoned = []
        for track_id, state in self.object_states.items():
            if state['class_name'] != 'person' and not state['alerted']:
                if state['stationary_frames'] > self.abandonment_threshold and state['history']:
                    is_abandoned = True
                    if person_centroids:
                        current_centroid = state['history'][-1]
                        for p_centroid in person_centroids:
                            dist = np.sqrt((current_centroid[0] - p_centroid[0])**2 + (current_centroid[1] - p_centroid[1])**2)
                            if dist < self.proximity_threshold:
                                is_abandoned = False
                                break
                    
                    if is_abandoned:
                        alerts.append(f"ALERT: Abandoned object (ID: {track_id}, Type: {state['class_name']}) detected!")
                        state['alerted'] = True # Fire alert only once
                        potential_abandoned.append(track_id)
        
        return alerts, potential_abandoned

    def check_loitering(self, loitering_zone_polygon):
        """
        Checks for loitering within a defined zone.
        """
        alerts = []
        loitering_ids = []
        
        if loitering_zone_polygon is None:
            return alerts, loitering_ids

        for track_id, state in self.object_states.items():
            if state['class_name'] == 'person' and not self.loitering_objects[track_id]['alerted']:
                if state['history']:
                    centroid = state['history'][-1]
                    # Check if the person's centroid is inside the polygon
                    if cv2.pointPolygonTest(loitering_zone_polygon, centroid, False) >= 0:
                        self.loitering_objects[track_id]['frames_in_zone'] += 1
                    else:
                        self.loitering_objects[track_id]['frames_in_zone'] = 0 # Reset if they leave the zone

                    if self.loitering_objects[track_id]['frames_in_zone'] > self.loitering_threshold:
                        alerts.append(f"ALERT: Loitering detected (ID: {track_id})!")
                        self.loitering_objects[track_id]['alerted'] = True
                        loitering_ids.append(track_id)
        
        return alerts, loitering_ids