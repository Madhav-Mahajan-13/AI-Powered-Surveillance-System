# drawing_utils.py
import cv2
import numpy as np

def draw_tracked_objects(frame, tracked_objects, class_names_map, logic_engine, abandoned_ids, loitering_ids):
    """
    Draw bounding boxes and information for tracked objects.
    """
    for obj in tracked_objects:
        x1, y1, x2, y2, track_id = map(int, obj)
        
        # Get object class name
        class_name = class_names_map.get(track_id, 'object')
        
        # Determine box color based on alert status
        color = (0, 255, 0)  # Green by default
        status = "Normal"
        
        if track_id in abandoned_ids:
            color = (0, 0, 255)  # Red for abandoned objects
            status = "ABANDONED"
        elif track_id in loitering_ids:
            color = (0, 165, 255)  # Orange for loitering
            status = "LOITERING"
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw label background
        label = f"ID:{track_id} {class_name} - {status}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
        
        # Draw label text
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Draw trajectory for tracked objects
        if track_id in logic_engine.object_states:
            history = logic_engine.object_states[track_id]['history']
            if len(history) > 1:
                points = np.array(history, dtype=np.int32)
                cv2.polylines(frame, [points], False, color, 1)
    
    return frame

def draw_loitering_zone(frame, loitering_zone):
    """
    Draw the loitering detection zone on the frame.
    """
    if loitering_zone is not None:
        # Draw the polygon
        cv2.polylines(frame, [loitering_zone], True, (255, 255, 0), 2)  # Yellow color
        
        # Add zone label
        zone_center = np.mean(loitering_zone, axis=0).astype(int)
        cv2.putText(frame, "LOITERING ZONE", (zone_center[0] - 60, zone_center[1]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    return frame

def draw_abandonment_highlight(frame, abandoned_objects, logic_engine):
    """
    Additional function to highlight abandoned objects with special effects.
    """
    for track_id in abandoned_objects:
        if track_id in logic_engine.object_states:
            state = logic_engine.object_states[track_id]
            if state['history']:
                centroid = state['history'][-1]
                # Draw pulsing circle around abandoned objects
                radius = 30 + int(10 * np.sin(state['stationary_frames'] * 0.1))
                cv2.circle(frame, centroid, radius, (0, 0, 255), 2)
    
    return frame

def draw_statistics(frame, logic_engine):
    """
    Draw statistics overlay on the frame.
    """
    # Count different object types
    total_objects = len(logic_engine.object_states)
    person_count = sum(1 for state in logic_engine.object_states.values() 
                      if state['class_name'] == 'person')
    abandoned_count = sum(1 for state in logic_engine.object_states.values() 
                         if state['alerted'] and state['class_name'] != 'person')
    
    # Draw statistics box
    stats_text = [
        f"Total Objects: {total_objects}",
        f"Persons: {person_count}",
        f"Abandoned Objects: {abandoned_count}"
    ]
    
    y_offset = 30
    for i, text in enumerate(stats_text):
        cv2.putText(frame, text, (10, y_offset + i * 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, text, (10, y_offset + i * 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    return frame