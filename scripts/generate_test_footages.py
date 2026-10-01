"""Create four short, upload-ready PPE test videos from the bundled samples."""

from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "sample_data"
FPS = 24
DURATION_SECONDS = 5


def animated_frame(image: np.ndarray, frame_number: int, direction: int) -> np.ndarray:
    """Create a gentle pan/zoom so the test file behaves like a real video."""
    height, width = image.shape[:2]
    progress = frame_number / (FPS * DURATION_SECONDS - 1)
    zoom = 1.0 + 0.06 * progress
    scaled = cv2.resize(image, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_LINEAR)
    scaled_height, scaled_width = scaled.shape[:2]
    max_x = max(0, scaled_width - width)
    max_y = max(0, scaled_height - height)
    x = int(max_x * (progress if direction > 0 else 1 - progress))
    y = int(max_y * (0.5 + 0.5 * np.sin(progress * np.pi)))
    return scaled[y:y + height, x:x + width]


def write_clip(source_name: str, output_name: str, direction: int) -> None:
    image = cv2.imread(str(SAMPLES / source_name))
    if image is None:
        raise FileNotFoundError(f"Could not read {source_name}")
    height, width = image.shape[:2]
    output = cv2.VideoWriter(
        str(SAMPLES / output_name), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (width, height)
    )
    for frame_number in range(FPS * DURATION_SECONDS):
        output.write(animated_frame(image, frame_number, direction))
    output.release()
    print(f"Created {output_name}")


for source, name, direction in (
    ("test_worker_sample.png", "helmet_test_pan_right_5s.mp4", 1),
    ("test_worker_sample.png", "helmet_test_pan_left_5s.mp4", -1),
    ("test_worker_no_helmet.png", "no_helmet_test_pan_right_5s.mp4", 1),
    ("test_worker_no_helmet.png", "no_helmet_test_pan_left_5s.mp4", -1),
):
    write_clip(source, name, direction)
