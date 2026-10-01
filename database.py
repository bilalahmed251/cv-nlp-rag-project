import sqlite3
import os
import cv2
import pandas as pd
from datetime import datetime
import time

DB_PATH = "incidents.db"
VIOLATIONS_DIR = "violations_screenshots"

# CONFIGURABLE GRACE PERIOD ALIGNED WITH BYTETRACK
# ByteTrack default track_buffer is 30 frames. It keeps a lost ID in memory for this duration.
# By aligning our grace period with this buffer, we ensure the incident remains OPEN 
# exactly as long as ByteTrack is still trying to recover the track.
TRACK_BUFFER_FRAMES = 30
VIDEO_FPS = 30 # Default assumed FPS
GRACE_PERIOD_SECONDS = TRACK_BUFFER_FRAMES / VIDEO_FPS # e.g., 1.0 seconds
# In-memory dictionary to track active violations
# Format: { "1_VIOLATION: Helmet Missing": {"db_id": 1, "start_epoch": ..., "last_seen_epoch": ..., "worker_id": 1} }
active_incidents = {}

def init_db():
    """Creates the folder and database table for continuous tracking."""
    os.makedirs(VIOLATIONS_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # Nayi table jisme requirement ke mutabiq exact columns hain
    c.execute('''
        CREATE TABLE IF NOT EXISTS continuous_incidents_v2 (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            worker_id INTEGER,
            violation_type TEXT,
            started_at DATETIME,
            last_seen_at DATETIME,
            ended_at DATETIME,
            duration_seconds INTEGER,
            incident_state TEXT,
            screenshot_path TEXT
        )
    ''')
    conn.commit()
    conn.close()

def process_worker_status(worker_id, status, frame_rgb):
    """
    Main function jo har frame par call hogi.
    Identifies incident by Worker ID + violation type.
    """
    current_time = time.time()
    is_violation = "VIOLATION" in status
    
    # Requirement: Identify incident by Worker ID + violation type
    incident_key = f"{worker_id}_{status}"
    
    if is_violation:
        if incident_key not in active_incidents:
            # 1. NAYI VIOLATION START HUI!
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"worker_{worker_id}_{timestamp_str}.jpg"
            filepath = os.path.join(VIOLATIONS_DIR, filename)
            
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(filepath, frame_bgr)
            
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            started_at_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            c.execute(
                "INSERT INTO continuous_incidents_v2 (worker_id, violation_type, started_at, last_seen_at, incident_state, screenshot_path) VALUES (?, ?, ?, ?, ?, ?)",
                (worker_id, status, started_at_str, started_at_str, 'OPEN', filepath)
            )
            db_id = c.lastrowid
            conn.commit()
            conn.close()
            
            # Memory mein yaad rakhein
            active_incidents[incident_key] = {
                "db_id": db_id,
                "start_epoch": current_time,
                "last_seen_epoch": current_time,
                "worker_id": worker_id
            }
            print(f"[OPEN] Incident {incident_key} started.")
        else:
            # 2. VIOLATION CHAL RAHI HAI (Continuous)
            active_incidents[incident_key]["last_seen_epoch"] = current_time
            
    else:
        # Worker is COMPLIANT. Close any open violations for this worker.
        keys_to_close = [k for k in active_incidents.keys() if active_incidents[k]["worker_id"] == worker_id]
        for k in keys_to_close:
            close_incident(k)

def close_incident(incident_key):
    """Database mein open incident ko update karke close karta hai."""
    incident = active_incidents.pop(incident_key, None)
    if not incident:
        return
        
    db_id = incident["db_id"]
    
    # Requirement: Accurate duration (calculated from last_seen, not current time!)
    last_seen = incident["last_seen_epoch"]
    duration = int(last_seen - incident["start_epoch"])
    
    # Format timestamps
    ended_at_str = datetime.fromtimestamp(last_seen).strftime("%Y-%m-%d %H:%M:%S")
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "UPDATE continuous_incidents_v2 SET ended_at = ?, last_seen_at = ?, duration_seconds = ?, incident_state = 'CLOSED' WHERE id = ?",
        (ended_at_str, ended_at_str, duration, db_id)
    )
    conn.commit()
    conn.close()
    print(f"[CLOSED] Incident {incident_key} closed. Duration: {duration}s")

def cleanup_ghosts():
    """Requirement: Close the incident only when violation has remained undetected for configurable grace period."""
    current_time = time.time()
    for incident_key in list(active_incidents.keys()):
        if current_time - active_incidents[incident_key]["last_seen_epoch"] > GRACE_PERIOD_SECONDS:
            print(f"[GHOST] Incident {incident_key} exceeded grace period. Closing.")
            close_incident(incident_key)

def get_recent_logs(limit=20):
    """Returns a pandas DataFrame of the most recent logs."""
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query(f"SELECT * FROM continuous_incidents_v2 ORDER BY started_at DESC LIMIT {limit}", conn)
    except:
        df = pd.DataFrame()
    conn.close()
    return df
