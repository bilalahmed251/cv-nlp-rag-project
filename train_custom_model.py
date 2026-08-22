"""
DEEP LEARNING TRAINING SCRIPT
-----------------------------
This script contains the actual Machine Learning and Deep Learning logic needed to 
TRAIN a custom model from scratch.

In a real scenario, you would run this code on Google Colab or Kaggle (because it requires a GPU).
"""

from ultralytics import YOLO

def train_custom_model():
    print("--- Starting Deep Learning Training Phase ---")
    
    # 1. INITIALIZE THE NEURAL NETWORK ARCHITECTURE
    # We load a blank or 'base' YOLO model. Think of this like a brain that knows 
    # the basics of "how to see shapes" but doesn't know any specific objects yet.
    # The 'n' in yolov8n.yaml means 'nano' (the smallest, fastest architecture).
    print("Loading Neural Network Architecture...")
    model = YOLO("yolov8n.yaml")  # .yaml loads the architecture, NOT the pre-trained weights
    
    # 2. DEFINE THE DATASET
    # In Deep Learning, your model is only as good as your data.
    # We will use Roboflow to download our Construction Safety dataset directly.
    print("Downloading dataset from Roboflow...")
    from roboflow import Roboflow
    rf = Roboflow(api_key="nMF4N1W4VedqxK0hDZx6")
    project = rf.workspace("new-workspace-0vfuy").project("construction-safety-hequn")
    version = project.version(6)
    dataset = version.download("yolov8")
    
    # 'data.yaml' is a file inside the downloaded dataset that tells the model:
    # - Where the training images are.
    # - Where the validation images are (used to test the model during training).
    # - What classes it needs to learn (e.g., 0: 'Hard Hat', 1: 'Safety Vest').
    dataset_path = f"{dataset.location}/data.yaml"
    
    # 3. THE TRAINING LOOP (The actual "Learning" part)
    # This is where the heavy Deep Learning math (Backpropagation, Gradient Descent) happens.
    print(f"Beginning training on dataset: {dataset_path}")
    
    results = model.train(
        data=dataset_path,
        
        # EPOCHS: How many times the neural network will see the ENTIRE dataset.
        # If it sees the data too few times, it won't learn (Underfitting).
        # If it sees it too many times, it memorizes the data instead of learning patterns (Overfitting).
        epochs=50,
        
        # BATCH SIZE: How many images the model looks at simultaneously before updating its weights.
        # Larger batches are faster but require more GPU memory.
        batch=16,
        
        # IMAGE SIZE: The resolution the images are resized to before being fed into the network.
        # 640x640 is standard for YOLO.
        imgsz=640,
        
        # LEARNING RATE (lr0): How big of a step the model takes when adjusting its mathematical weights.
        # Too big: It jumps over the optimal solution.
        # Too small: It takes way too long to learn.
        lr0=0.01,
        
        # DEVICE: Tells the code to run on a GPU ('0') instead of a CPU ('cpu') for speed.
        device="0" 
    )
    
    print("--- Training Complete! ---")
    print("The model has now learned the patterns and saved its 'brain' as a new .pt file (usually in runs/detect/train/weights/best.pt).")
    print("We can now take THIS custom .pt file and put it into our cv_engine.py!")

if __name__ == "__main__":
    print("WARNING: This is the Deep Learning training logic.")
    print("To actually run this, you need a dataset and a GPU (like Google Colab).")
    print("Read the comments in this file to understand the ML math!")
