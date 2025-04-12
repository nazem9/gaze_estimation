import os
import torch
import cv2
import numpy as np
import json
from tqdm import tqdm
from ultralytics import YOLO  # Assuming you use the ultralytics YOLO package
# Assuming the gaze pipeline code is in a file named gaze_pipeline.py
from gaze_pipeline import Pipeline, GazeResultContainer 

# --- Configuration ---
DATASET_IMAGE_DIR = "/path/to/your/image/dataset" # Directory containing images
OUTPUT_ANNOTATION_FILE = "distillation_annotations.json"
PHONE_MODEL_PATH = 'yolov8l.pt' # Or your specific yolo model
L2CS_MODEL_PATH = '/path/to/your/l2cs_model.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMAGE_SIZE_FOR_TEACHERS = 640 # Example size, adjust if needed for YOLO/RetinaFace
FACE_CONF_THRESHOLD = 0.5
PHONE_CONF_THRESHOLD = 0.5
PHONE_CLASS_ID = 67 # Common class ID for mobile phone in COCO

# --- Initialize Teacher Models ---
print("Loading teacher models...")
phone_detector = YOLO(PHONE_MODEL_PATH)
gaze_pipeline = Pipeline(
    weights=L2CS_MODEL_PATH,
    arch='ResNet50', # Or the correct architecture
    device=DEVICE,
    include_detector=True, # Use RetinaFace within the pipeline
    confidence_threshold=FACE_CONF_THRESHOLD
)
print("Teacher models loaded.")

# --- Process Dataset ---
all_annotations = []
image_files = [f for f in os.listdir(DATASET_IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

print(f"Processing {len(image_files)} images...")
for img_name in tqdm(image_files):
    img_path = os.path.join(DATASET_IMAGE_DIR, img_name)
    try:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read image {img_path}")
            continue

        img_h, img_w = img.shape[:2]

        # --- 1. Get Face, Landmark, Gaze Predictions ---
        # Gaze pipeline expects BGR
        gaze_results: GazeResultContainer = gaze_pipeline.step(img.copy()) # Pass a copy

        faces_data = []
        if gaze_results.bboxes.shape[0] > 0:
            for i in range(gaze_results.bboxes.shape[0]):
                # Normalize coordinates to [0, 1]
                x1, y1, x2, y2 = gaze_results.bboxes[i]
                norm_box = [x1 / img_w, y1 / img_h, x2 / img_w, y2 / img_h]

                # Normalize landmarks
                # RetinaFace landmarks are [l_eye_x, l_eye_y, r_eye_x, r_eye_y, nose_x, nose_y, l_mouth_x, l_mouth_y, r_mouth_x, r_mouth_y]
                # Your SSD expects 68. We cannot directly map 5 -> 68.
                # Options:
                #   a) Train a separate small model to predict 68 landmarks from the 5 + face crop.
                #   b) Use only the 5 landmarks (requires changing SSD head/loss).
                #   c) Omit landmarks during distillation if 68 are not essential initially.
                # For now, we'll store the 5 landmarks and handle the mismatch later or ignore.
                norm_landmarks_5 = gaze_results.landmarks[i].flatten() / np.tile([img_w, img_h], 5)

                # Placeholder for 68 landmarks - NEEDS A SOLUTION
                norm_landmarks_68 = np.zeros(136) # Or use the 5? Or skip?

                faces_data.append({
                    "box": norm_box,
                    "score": float(gaze_results.scores[i]),
                    "landmarks_5": norm_landmarks_5.tolist(),
                    "landmarks_68_placeholder": norm_landmarks_68.tolist(), # Mark as placeholder
                    "gaze": [float(gaze_results.pitch[i]), float(gaze_results.yaw[i])] # Pitch, Yaw
                })

        # --- 2. Get Phone Predictions ---
        # YOLO expects RGB and possibly resizing/normalization (check YOLO docs)
        # Often takes PIL images or numpy arrays directly. Let's use the path.
        phone_preds = phone_detector.predict(img_path, classes=[PHONE_CLASS_ID], conf=PHONE_CONF_THRESHOLD, device=DEVICE)

        phones_data = []
        # Assuming phone_preds[0] contains the results for the first image
        if len(phone_preds) > 0 and phone_preds[0].boxes is not None:
            for box in phone_preds[0].boxes:
                 # Get coordinates (xyxyn format is normalized)
                norm_box = box.xyxyn[0].cpu().numpy().tolist()
                score = float(box.conf[0].cpu().numpy())
                phones_data.append({
                    "box": norm_box,
                    "score": score
                })

        # --- 3. Store Combined Annotations ---
        if faces_data or phones_data: # Only save if we found something
            all_annotations.append({
                "image_path": img_path, # Store relative or absolute path
                "image_height": img_h,
                "image_width": img_w,
                "faces": faces_data,
                "phones": phones_data
            })

    except Exception as e:
        print(f"Error processing {img_name}: {e}")

# --- Save Annotations ---
print(f"Saving {len(all_annotations)} annotations to {OUTPUT_ANNOTATION_FILE}...")
with open(OUTPUT_ANNOTATION_FILE, 'w') as f:
    json.dump(all_annotations, f, indent=2)

print("Annotation generation complete.")