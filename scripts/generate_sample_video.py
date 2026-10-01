import cv2
import numpy as np

# Load the existing sample image
img = cv2.imread('test_worker_sample.png')

if img is None:
    print("Error: Could not read test_worker_sample.png")
else:
    height, width, layers = img.shape
    
    # Create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('test_video_sample.mp4', fourcc, 30.0, (width, height))
    
    print("Generating a 3-second sample video...")
    
    # Generate 90 frames (3 seconds at 30 fps)
    for i in range(90):
        frame = img.copy()
        
        # Add a moving "LIVE" indicator or timestamp to prove it's a video
        time_text = f"LIVE FEED: 00:00:{i//30:02d}"
        cv2.putText(frame, time_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        
        # Draw a moving dot
        dot_x = 50 + (i * 5)
        cv2.circle(frame, (dot_x, 80), 10, (0, 0, 255), -1)
        
        out.write(frame)
        
    out.release()
    print("Video 'test_video_sample.mp4' created successfully in your folder!")
