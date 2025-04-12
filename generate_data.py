import cv2
import numpy as np
import torch
import pathlib
import json
from tqdm import tqdm
from l2cs import Pipeline
from l2cs.results import GazeResultContainer
from ultralytics import YOLO

BASE_IMAGE_DIR = pathlib.Path("./dataset")
ANNOTATION_OUTPUT_DIR = pathlib.Path("./generated_annotations_ssd")
L2CS_MODEL_PATH = pathlib.Path("l2cs_model/L2CSNet_gaze360.pkl")
YOLO_MODEL_PATH = "yolo11l.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GAZE_CONFIDENCE_THRESHOLD = 0.8
PHONE_CONFIDENCE_THRESHOLD = 0.5
IMG_EXTENSIONS = [".jpg", ".jpeg", ".png"]
CLASS_MAP = {
    'face': 0,
    'phone':1
}

gaze_pipeline = Pipeline(
    weights=L2CS_MODEL_PATH,
    arch="ResNet50",
    device=DEVICE,
    include_detector=True,
    confidence_threshold=GAZE_CONFIDENCE_THRESHOLD
)

phone_predictor = YOLO(YOLO_MODEL_PATH)

def generate_annotations(image_dir, output_dir):
    image_files = [p for p in image_dir.rglob("*") if p.suffix.lower() in IMG_EXTENSIONS]
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Processing {len(image_files)} images in {image_dir}")

    for img_path in tqdm(image_files):
        try:
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                continue

            height, width, _ = img_bgr.shape

            phone_results = phone_predictor(img_bgr, classes=[67], conf=PHONE_CONFIDENCE_THRESHOLD, verbose=False)
            phone_annotations = []
            for result in phone_results:
                for box in result.boxes:
                    xyxy = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().item()
                    cls_id = int(box.cls[0].cpu().item())
                    if cls_id == 67:
                        phone_annotations.append({
                            "bbox": [float(c) for c in xyxy],
                            "class_id": CLASS_MAP["phone"],
                            "confidence": float(conf),
                            "landmarks": None,
                            "gaze": None
                        })

            try:
                gaze_results: GazeResultContainer = gaze_pipeline.step(img_bgr.copy())
            except Exception as e:
                if "need at least one array to stack" in str(e):
                    # Create an empty gaze result container instead of failing
                    gaze_results = GazeResultContainer(
                        bboxes=[],
                        landmarks=[],
                        scores=[],
                        pitch=[],
                        yaw=[]
                    )
                else:
                    raise e

            face_annotations = []
            if gaze_results.scores is not None and len(gaze_results.scores) > 0:
                num_faces = len(gaze_results.scores)
                for i in range(num_faces):
                    if i < len(gaze_results.bboxes) and \
                       i < len(gaze_results.landmarks) and \
                       i < len(gaze_results.pitch) and \
                       i < len(gaze_results.yaw):
                        bbox = gaze_results.bboxes[i]
                        landmark = gaze_results.landmarks[i]
                        score = gaze_results.scores[i]
                        pitch = gaze_results.pitch[i]
                        yaw = gaze_results.yaw[i]
                        face_annotations.append({
                            "bbox": [float(c) for c in bbox],
                            "class_id": CLASS_MAP["face"],
                            "confidence": float(score),
                            "landmarks": landmark.tolist(),
                            "gaze": [float(pitch), float(yaw)]
                        })

            all_annotations = phone_annotations + face_annotations
            if not all_annotations:
                continue

            output_data = {
                "image_path": str(img_path.relative_to(image_dir)),
                "image_height": height,
                "image_width": width,
                "annotations": all_annotations
            }

            output_json_path = output_dir / (img_path.stem + ".json")
            with open(output_json_path, "w") as f:
                json.dump(output_data, f, indent=2)

        except Exception as e:
            print(f"Error processing image {img_path}: {e}")

for split in ["train"]:
    image_dir = BASE_IMAGE_DIR / split / "images"
    output_dir = ANNOTATION_OUTPUT_DIR / split
    generate_annotations(image_dir, output_dir)

print("Annotation generation complete.")
