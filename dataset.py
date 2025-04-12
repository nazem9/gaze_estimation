# dataset.py
import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image
import numpy as np
from config import ANNOTATION_DIR, IMAGE_BASE_DIR, CLASS_MAP, NUM_LANDMARKS

class CombinedDataset(Dataset):
    def __init__(self, annotation_dir=ANNOTATION_DIR, image_base_dir=IMAGE_BASE_DIR, transform=None, img_size=300):
        self.annotation_dir = annotation_dir
        self.image_base_dir = image_base_dir
        self.transform = transform
        self.img_size = img_size
        self.json_files = [f for f in os.listdir(annotation_dir) if f.endswith('.json')]
        self.face_class_id = CLASS_MAP['face']

    def __len__(self):
        return len(self.json_files)

    def __getitem__(self, idx):
        json_path = os.path.join(self.annotation_dir, self.json_files[idx])

        with open(json_path, 'r') as f:
            data = json.load(f)

        img_rel_path = data['image_path']
        img_abs_path = os.path.join(self.image_base_dir, img_rel_path)
        # Use PIL for compatibility with torchvision transforms
        try:
             image = Image.open(img_abs_path).convert('RGB')
             img_w, img_h = image.size
        except FileNotFoundError:
             print(f"Warning: Image not found {img_abs_path}. Skipping.")
             # Handle appropriately - maybe return None or raise error?
             # For now, return dummy data for this index
             return self.__getitem__((idx + 1) % len(self)) # Get next item


        boxes = []
        labels = []
        landmarks = [] # List to collect landmarks only for faces
        gaze_data = [] # List to collect gaze only for faces

        num_faces = 0
        for ann in data['annotations']:
            # --- Bounding Box ---
            # Annotation format: [xmin, ymin, xmax, ymax] (absolute pixels)
            # Normalize to 0-1 range
            xmin, ymin, xmax, ymax = ann['bbox']
            boxes.append([xmin / img_w, ymin / img_h, xmax / img_w, ymax / img_h])

            # --- Label ---
            # Annotation format: class_id (0=face, 1=phone) -> map to config (1=face, 2=phone)
            # The loaded class_id from JSON needs mapping to CLASS_MAP values
            # Assuming JSON stores 0 for face, 1 for phone as in generation script
            json_class_id = ann['class_id']
            if json_class_id == 0: # Face from JSON
                label = CLASS_MAP['face']
            elif json_class_id == 1: # Phone from JSON
                label = CLASS_MAP['phone']
            else:
                 # Handle unexpected class ID if necessary
                 print(f"Warning: Unexpected class ID {json_class_id} in {json_path}")
                 label = 0 # Treat as background or skip? Needs decision. Assigning bg for now.
            labels.append(label)

            # --- Landmarks & Gaze (Conditional) ---
            if label == self.face_class_id:
                num_faces += 1
                # Landmarks: (5, 2) list -> flatten to 10, normalize
                lmk_list = ann['landmarks'] # List of [x, y] pairs
                lmk_flat_norm = []
                if lmk_list:
                   for x, y in lmk_list:
                       lmk_flat_norm.extend([x / img_w, y / img_h])
                else:
                     # Handle missing landmarks if possible (e.g., fill with zeros, but model should handle)
                     lmk_flat_norm = [0.0] * (NUM_LANDMARKS * 2)

                landmarks.append(lmk_flat_norm)

                # Gaze: [pitch, yaw] list (assume radians)
                gaze_list = ann['gaze']
                if gaze_list:
                    gaze_data.append(gaze_list)
                else:
                    # Handle missing gaze
                    gaze_data.append([0.0, 0.0])


        # Convert to tensors
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)
        landmarks = torch.tensor(landmarks, dtype=torch.float32) # (N_faces, 10)
        gaze_data = torch.tensor(gaze_data, dtype=torch.float32) # (N_faces, 2)

        # Create target dictionary
        target = {
            'boxes': boxes,       # (N_obj, 4) xyxy normalized
            'labels': labels,     # (N_obj) class indices (1=face, 2=phone)
        }
        # Only add landmarks/gaze if faces were present
        if num_faces > 0:
            target['landmarks'] = landmarks # (N_faces, 10) normalized
            target['gaze'] = gaze_data       # (N_faces, 2) radians

        # Apply transformations (e.g., resizing, normalization, data augmentation)
        if self.transform:
            # Note: Transforms need to handle both image and targets correctly!
            # Standard torchvision transforms usually only handle the image.
            # You might need custom transforms or libraries like Albumentations
            # that can transform images and bounding boxes/landmarks together.
            # Placeholder: applying transform only to image
            image = self.transform(image)
             # TODO: Ensure bounding boxes/landmarks are adjusted if geometric transforms are used

        return image, target

# --- Collate Function (important for handling batches with varying numbers of objects) ---
def collate_fn(batch):
    images = []
    targets = []
    for img, tgt in batch:
        if img is not None: # Handle potential loading errors from dataset
             images.append(img)
             targets.append(tgt)
    if not images: return None, None # Handle empty batch
    # Stack images (assuming they are tensors of same size after transform)
    images = torch.stack(images, dim=0)
    return images, targets # Return list of target dicts