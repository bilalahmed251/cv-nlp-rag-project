import cv2
from ultralytics import YOLO

class CVEngine:
    def __init__(self, model_name='best.pt'):
        """
        Initializes the Computer Vision engine with your custom trained YOLOv8 model.
        'best.pt' contains the specific knowledge of helmets vs no_helmets.
        """
        print(f"Loading {model_name}...")
        self.model = YOLO(model_name)

    def analyze_image(self, image_path, conf_threshold=0.15):
        """
        Takes an image path, runs object detection with a high-recall confidence threshold,
        and returns a summary of findings along with the annotated image array.
        """
        print(f"Analyzing {image_path} with confidence threshold {conf_threshold}...")
        
        # Run inference with confidence threshold 0.15 for high sensitivity
        results = self.model(image_path, conf=conf_threshold)
        
        annotated_img_rgb = None
        if len(results) > 0:
            # plot() returns a BGR numpy array containing bounding boxes
            annotated_img_bgr = results[0].plot()
            # Convert BGR to RGB for Streamlit/PIL display
            annotated_img_rgb = annotated_img_bgr[..., ::-1]
            
        # Parse the results
        detections = self._parse_and_associate_detections(results)
        
        return detections, annotated_img_rgb

    @staticmethod
    def _box_center(box):
        """Return the centre point of an xyxy bounding box."""
        x1, y1, x2, y2 = box
        return (x1 + x2) / 2, (y1 + y2) / 2

    def _find_matching_gear(self, person, gear_items, used_indexes):
        """Find the closest unused PPE detection in the upper area of one person."""
        px1, py1, px2, py2 = person["box"]
        upper_body_limit = py1 + (py2 - py1) * 0.55
        person_center_x = (px1 + px2) / 2

        candidates = []
        for index, gear in enumerate(gear_items):
            if index in used_indexes:
                continue
            gear_center_x, gear_center_y = self._box_center(gear["box"])
            if px1 <= gear_center_x <= px2 and py1 <= gear_center_y <= upper_body_limit:
                candidates.append((abs(gear_center_x - person_center_x), index, gear))

        if not candidates:
            return None, None

        _, index, gear = min(candidates, key=lambda candidate: candidate[0])
        return index, gear

    def _build_compliance_data(self, results):
        """Convert YOLO boxes into display detections and one PPE status per worker."""
        """
        Associate helmet detections with person boxes.  When tracking is enabled,
        Ultralytics supplies a stable ``track_id`` for each worker across frames.
        """
        detections_list = []
        persons = []
        helmets = []
        no_helmets = []

        if len(results) == 0:
            return [], []

        for box in results[0].boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = self.model.names[class_id]
            
            # Get bounding box coordinates [x1, y1, x2, y2]
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            track_id = int(box.id[0]) if box.id is not None else None
            det_info = {
                "class": class_name,
                "conf": confidence,
                "box": (x1, y1, x2, y2),
                "track_id": track_id,
            }
            
            if class_name == "person":
                persons.append(det_info)
            elif class_name == "helmet":
                helmets.append(det_info)
            elif class_name == "no_helmet" or class_name == "no-helmet":
                no_helmets.append(det_info)
            else:
                detections_list.append({"object": class_name, "confidence": round(confidence * 100, 2)})

        workers = []

        # A gear box must be assigned to at most one person in a frame.
        used_helmets = set()
        used_no_helmets = set()

        if persons:
            for worker_number, person in enumerate(persons, start=1):
                # Safety-first rule: an explicit no-helmet detection takes priority.
                no_helmet_index, no_helmet = self._find_matching_gear(
                    person, no_helmets, used_no_helmets
                )
                helmet_index, helmet = self._find_matching_gear(person, helmets, used_helmets)

                worker_id = person["track_id"] if person["track_id"] is not None else worker_number
                worker = {
                    "worker_id": worker_id,
                    "person_confidence": round(person["conf"] * 100, 2),
                    "box": person["box"],
                }

                if no_helmet:
                    used_no_helmets.add(no_helmet_index)
                    worker.update({
                        "status": "VIOLATION: Helmet Missing",
                        "equipment_confidence": round(no_helmet["conf"] * 100, 2),
                    })
                elif helmet:
                    used_helmets.add(helmet_index)
                    worker.update({
                        "status": "COMPLIANT: Helmet Detected",
                        "equipment_confidence": round(helmet["conf"] * 100, 2),
                    })
                else:
                    worker.update({
                        "status": "UNKNOWN: Helmet Not Verified",
                        "equipment_confidence": None,
                    })

                workers.append(worker)
                detections_list.append({
                    "object": f"Worker {worker_id} - {worker['status']}",
                    "confidence": worker["equipment_confidence"] or worker["person_confidence"],
                })
        else:
            # Fallback if model only detects helmet/no_helmet as heads (no person box)
            for idx, h in enumerate(helmets, start=1):
                detections_list.append({"object": "Worker with Helmet", "confidence": round(h["conf"] * 100, 2)})
                workers.append({
                    "worker_id": h.get("track_id") or f"H{idx}",
                    "status": "COMPLIANT: Helmet Detected",
                    "equipment_confidence": round(h["conf"] * 100, 2),
                    "box": h["box"]
                })
            for idx, nh in enumerate(no_helmets, start=1):
                detections_list.append({"object": "Worker WITHOUT Helmet", "confidence": round(nh["conf"] * 100, 2)})
                workers.append({
                    "worker_id": nh.get("track_id") or f"NH{idx}",
                    "status": "VIOLATION: Helmet Missing",
                    "equipment_confidence": round(nh["conf"] * 100, 2),
                    "box": nh["box"]
                })
                
        return detections_list, workers

    def _parse_and_associate_detections(self, results):
        """Backwards-compatible summary used by image and uploaded-video analysis."""
        detections, _ = self._build_compliance_data(results)
        return detections

    def analyze_live_frame(self, frame_bgr, conf_threshold=0.45):
        """Track workers in one webcam frame and return an annotated frame plus PPE status."""
        results = self.model.track(
            frame_bgr,
            conf=conf_threshold,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
        )
        annotated_frame = results[0].plot() if results else frame_bgr.copy()
        _, workers = self._build_compliance_data(results)

        for worker in workers:
            x1, y1, _, _ = map(int, worker["box"])
            is_violation = worker["status"].startswith("VIOLATION")
            color = (0, 0, 255) if is_violation else (0, 180, 0)
            label = f"Worker {worker['worker_id']}: {worker['status'].split(': ', 1)[-1]}"
            cv2.putText(
                annotated_frame,
                label,
                (x1, max(25, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

        return annotated_frame, workers

    def analyze_video(self, video_path, conf_threshold=0.15, frame_skip=5):
        """
        Reads a video file frame-by-frame, runs object detection, and yields 
        the annotated RGB frame along with cumulative detections so far and progress.
        """
        print(f"Analyzing video {video_path} with confidence threshold {conf_threshold}...")
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return
            
        cumulative_detections_map = {}
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break # End of video
                
            frame_count += 1
            if frame_count % frame_skip != 0:
                continue
                
            # Resize frame to max width 640 to speed up processing
            h, w = frame.shape[:2]
            if w > 640:
                scale = 640 / w
                frame = cv2.resize(frame, (640, int(h * scale)))

            # Use ByteTrack for stable worker IDs across video frames
            results = self.model.track(
                frame, 
                conf=conf_threshold, 
                persist=True, 
                tracker="bytetrack.yaml", 
                verbose=False
            )
            
            annotated_frame_rgb = None
            workers = []
            if len(results) > 0:
                annotated_frame_bgr = results[0].plot()
                annotated_frame_rgb = annotated_frame_bgr[..., ::-1] # Convert BGR to RGB
                
                current_dets, workers = self._build_compliance_data(results)
                for det in current_dets:
                    cumulative_detections_map[det["object"]] = det
                    
            # Convert map to list
            current_detections = list(cumulative_detections_map.values())
            progress = min(1.0, frame_count / total_frames) if total_frames > 0 else 0.0
            
            yield annotated_frame_rgb, current_detections, workers, progress
            
        cap.release()

# --- Test the engine if run directly ---
if __name__ == "__main__":
    engine = CVEngine()
    print("\n--- Testing CV Engine ---")
    results, img = engine.analyze_image("test_worker_sample.png")
    print(f"Found {len(results)} objects:")
    for r in results:
        print(f"- {r['object']}: {r['confidence']}%")
