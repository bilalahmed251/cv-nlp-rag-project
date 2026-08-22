import cv2
from ultralytics import YOLO
from PIL import Image

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
        and returns a summary of findings.
        """
        print(f"Analyzing {image_path} with confidence threshold {conf_threshold}...")
        
        # Run inference with confidence threshold 0.15 for high sensitivity
        results = self.model(image_path, conf=conf_threshold)
        
        # Parse the results
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get the class ID and confidence score
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                
                # Convert class ID to actual object name
                class_name = self.model.names[class_id]
                
                detections.append({
                    "object": class_name,
                    "confidence": round(confidence * 100, 2)
                })
        
        return detections

# --- Test the engine if run directly ---
if __name__ == "__main__":
    engine = CVEngine()
    print("\n--- Testing CV Engine ---")
    results = engine.analyze_image("test_worker_sample.png")
    print(f"Found {len(results)} objects:")
    for r in results:
        print(f"- {r['object']}: {r['confidence']}%")
