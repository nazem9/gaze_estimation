import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import resnet50, ResNet50_Weights
from typing import List, Tuple, Dict
import numpy as np

class SSDHead(nn.Module):
    """
    SSD detection head for multiple tasks: faces, landmarks, gaze, and phones.
    Now handles multi-class classification.
    """
    def __init__(self, in_channels: int, num_anchors: int, num_classes: int):
        super(SSDHead, self).__init__()
        self.num_classes = num_classes

        # Shared Box Regression (predicts offsets for any object type)
        self.box_conv = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=3, padding=1)

        # Class prediction (Background, Face, Phone)
        self.class_conv = nn.Conv2d(in_channels, num_anchors * num_classes, kernel_size=3, padding=1)

        # Facial landmarks (only relevant for face class)
        # Output size might depend on how you solved the landmark mismatch
        self.landmarks_conv = nn.Conv2d(in_channels, num_anchors * 136, kernel_size=3, padding=1) # Assuming 68 landmarks

        # Gaze estimation (only relevant for face class)
        self.gaze_conv = nn.Conv2d(in_channels, num_anchors * 2, kernel_size=3, padding=1)

    def forward(self, x):
        batch_size = x.size(0)

        # Box regression predictions
        box_preds = self.box_conv(x)
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 4)

        # Class predictions
        class_preds = self.class_conv(x)
        class_preds = class_preds.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, self.num_classes)

        # Landmark predictions
        landmark_preds = self.landmarks_conv(x)
        landmark_preds = landmark_preds.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 136) # Adjust size if needed

        # Gaze predictions
        gaze_preds = self.gaze_conv(x)
        gaze_preds = gaze_preds.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2)

        return class_preds, box_preds, landmark_preds, gaze_preds


class SSDFeatureExtractor(nn.Module):
    """
    Feature extractor using ResNet backbone with additional SSD feature layers
    """
    def __init__(self):
        super(SSDFeatureExtractor, self).__init__()
        # Load pretrained ResNet50
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        
        # Extract feature layers from ResNet
        self.from_resnet = nn.ModuleList([
            nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool),
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        ])
        
        # Additional SSD feature layers with decreasing spatial dimensions
        self.extras = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(2048, 256, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(512, 128, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True)
            )
        ])
        
    def forward(self, x):
        # Extract features from ResNet backbone
        feature_maps = []
        
        for i, layer in enumerate(self.from_resnet):
            x = layer(x)
            if i >= 2:  # Collect feature maps from layer2, layer3, and layer4
                feature_maps.append(x)
        
        # Apply additional SSD feature layers
        for layer in self.extras:
            x = layer(x)
            feature_maps.append(x)
            
        return feature_maps

class AnchorGenerator:
    """
    Generate anchor boxes for SSD model
    """
    def __init__(self, feature_map_sizes: List[Tuple[int, int]], 
                 min_sizes: List[int], 
                 max_sizes: List[int], 
                 aspect_ratios: List[List[float]],
                 image_size: int = 300):
        self.feature_map_sizes = feature_map_sizes
        self.min_sizes = min_sizes
        self.max_sizes = max_sizes
        self.aspect_ratios = aspect_ratios
        self.image_size = image_size
        
    def generate_anchors(self):
        """Generate anchor boxes for all feature maps"""
        anchors = []
        
        for k, feature_size in enumerate(self.feature_map_sizes):
            # Generate grid centers
            grid_height, grid_width = feature_size
            scale = self.image_size / min(self.image_size, 300)
            
            for i in range(grid_height):
                for j in range(grid_width):
                    # Calculate center positions
                    cx = (j + 0.5) / grid_width
                    cy = (i + 0.5) / grid_height
                    
                    # Small sized anchors
                    min_size = self.min_sizes[k] * scale
                    h = w = min_size / self.image_size
                    anchors.append([cx, cy, w, h])
                    
                    # Large sized anchors
                    if self.max_sizes[k] > 0:
                        max_size = self.max_sizes[k] * scale
                        h = w = max_size / self.image_size
                        anchors.append([cx, cy, w, h])
                    
                    # Anchors with different aspect ratios
                    min_size = self.min_sizes[k] * scale
                    for ratio in self.aspect_ratios[k]:
                        w = min_size * np.sqrt(ratio) / self.image_size
                        h = min_size / np.sqrt(ratio) / self.image_size
                        anchors.append([cx, cy, w, h])
        
        return torch.tensor(anchors)

class SSDResNetFaceLandmarkGaze(nn.Module):
    """
    Complete SSD model with ResNet backbone for face, landmark, gaze, and phone detection.
    """
    # Add PHONE class
    def __init__(self, num_classes: int = 3, image_size: int = 300): # num_classes = Background, Face, Phone
        super(SSDResNetFaceLandmarkGaze, self).__init__()
        self.image_size = image_size
        self.num_classes = num_classes

        # Feature extractor
        self.feature_extractor = SSDFeatureExtractor()

        # --- Define feature map sizes, anchor configs (adjust if feature extractor changed) ---
        # Example channels corresponding to the feature maps from SSDFeatureExtractor
        # Check the actual output channels of your `feature_extractor`
        # Example: Output channels might be [512, 1024, 2048, 512, 256, 256] if using ResNet layer outputs directly
        # Need to confirm these match the layers in SSDFeatureExtractor
        # Assuming the extractor outputs features compatible with these sizes/channels
        channels = [512, 1024, 512, 256, 256, 256] # Example channels, VERIFY THESE!
        # Feature map sizes depend on image_size and feature_extractor strides
        # For image_size=300 and default ResNet/SSD strides:
        self.feature_map_sizes = [(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)] # Example, VERIFY!
        # --- Anchor configuration ---
        self.min_sizes = [30, 60, 111, 162, 213, 264]
        self.max_sizes = [60, 111, 162, 213, 264, 315]
        self.aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
        self.num_anchors_per_cell = [len(ratios) * 2 + 2 for ratios in self.aspect_ratios] # Adjust based on min/max/ratios used

        # Check consistency
        if len(channels) != len(self.feature_map_sizes) or \
           len(self.num_anchors_per_cell) != len(self.feature_map_sizes):
             raise ValueError("Mismatch in feature map, channel, or anchor config lengths.")

        # --- Generate anchors ---
        self.anchor_generator = AnchorGenerator(
            self.feature_map_sizes,
            self.min_sizes,
            self.max_sizes,
            self.aspect_ratios,
            self.image_size
        )
        self.anchors = self.anchor_generator.generate_anchors() # Shape: [total_num_anchors, 4]
        self.anchors = nn.Parameter(self.anchors, requires_grad=False) # Store as non-trainable parameter


        # Create detection heads for each feature map
        self.detection_heads = nn.ModuleList([
            SSDHead(channels[i], self.num_anchors_per_cell[i], self.num_classes)
            for i in range(len(self.feature_map_sizes))
        ])

    def forward(self, x):
        # Extract features
        # Ensure feature_extractor returns the expected number of feature maps
        feature_maps = self.feature_extractor(x)
        if len(feature_maps) != len(self.detection_heads):
             raise RuntimeError(f"Feature extractor returned {len(feature_maps)} maps, expected {len(self.detection_heads)}")


        # Apply detection heads
        class_preds_list = []
        box_preds_list = []
        landmark_preds_list = []
        gaze_preds_list = []

        for i, feature in enumerate(feature_maps):
            class_preds, box_preds, landmark_preds, gaze_preds = self.detection_heads[i](feature)
            class_preds_list.append(class_preds)
            box_preds_list.append(box_preds)
            landmark_preds_list.append(landmark_preds) # Collect even if not trained, for structure
            gaze_preds_list.append(gaze_preds) # Collect even if not trained, for structure

        # Concatenate predictions from different feature maps
        class_preds = torch.cat(class_preds_list, dim=1)
        box_preds = torch.cat(box_preds_list, dim=1)
        landmark_preds = torch.cat(landmark_preds_list, dim=1)
        gaze_preds = torch.cat(gaze_preds_list, dim=1)

        return {
            'class_preds': class_preds,    # Shape: [batch_size, num_anchors, num_classes]
            'box_preds': box_preds,        # Shape: [batch_size, num_anchors, 4]
            'landmark_preds': landmark_preds,# Shape: [batch_size, num_anchors, 136]
            'gaze_preds': gaze_preds,      # Shape: [batch_size, num_anchors, 2]
            'anchors': self.anchors        # Shape: [num_anchors, 4] (cx, cy, w, h)
        }

    def _convert_boxes(self, box_preds, anchors):
        """ Convert predicted box offsets (cx_off, cy_off, w_log, h_log) to absolute coords (x1, y1, x2, y2)"""
        # Decode bounding box predictions relative to anchors
        # Variances often used in SSD: [0.1, 0.1, 0.2, 0.2]
        # box_preds are typically: dx, dy, dw, dh
        # cx = anchor_cx + dx * anchor_w * variance[0]
        # cy = anchor_cy + dy * anchor_h * variance[1]
        # w = anchor_w * exp(dw * variance[2])
        # h = anchor_h * exp(dh * variance[3])
        var = torch.tensor([0.1, 0.1, 0.2, 0.2], device=anchors.device)

        cxcy = box_preds[:, :2] * var[0] * anchors[:, 2:] + anchors[:, :2] # Center coords
        wh = torch.exp(box_preds[:, 2:] * var[2]) * anchors[:, 2:]       # Width/Height

        # Convert to [x1, y1, x2, y2] format
        boxes_xyxy = torch.cat([
            cxcy - wh / 2,  # x1, y1
            cxcy + wh / 2   # x2, y2
        ], dim=1)

        # Clip boxes to image boundaries [0, 1]
        boxes_xyxy = torch.clamp(boxes_xyxy, 0, 1)
        return boxes_xyxy

    def predict(self, x, confidence_threshold=0.05, top_k=200, nms_threshold=0.45):
        """ Perform prediction with post-processing for multi-class """
        self.eval() # Set model to evaluation mode
        with torch.no_grad():
            # Forward pass
            output = self(x)

            # Get raw predictions
            class_preds = output['class_preds'] # [B, num_anchors, num_classes]
            box_preds = output['box_preds']     # [B, num_anchors, 4]
            landmark_preds = output['landmark_preds'] # [B, num_anchors, 136]
            gaze_preds = output['gaze_preds']       # [B, num_anchors, 2]
            anchors = output['anchors'].to(x.device)  # [num_anchors, 4]

            batch_size = class_preds.size(0)
            num_anchors = anchors.size(0)
            num_classes = class_preds.size(2)

            # Apply softmax to class scores (excluding background class 0)
            class_scores = F.softmax(class_preds, dim=2)[:, :, 1:] # Scores for actual classes (Face, Phone)

            # Decode boxes for all anchors
            decoded_boxes = self._convert_boxes(box_preds.view(-1, 4), anchors.repeat(batch_size, 1)).view(batch_size, -1, 4)

            # Process each image in the batch
            results = []
            for i in range(batch_size):
                img_scores = class_scores[i] # [num_anchors, num_classes-1]
                img_boxes = decoded_boxes[i] # [num_anchors, 4]
                img_landmarks = landmark_preds[i] # [num_anchors, 136]
                img_gaze = gaze_preds[i]         # [num_anchors, 2]

                final_boxes = []
                final_scores = []
                final_labels = []
                final_landmarks = []
                final_gaze = []

                # Iterate through each class (Face=0, Phone=1 in scores tensor)
                for class_idx in range(img_scores.size(1)):
                    actual_class_label = class_idx + 1 # Map back to original label (1=Face, 2=Phone)
                    cls_scores = img_scores[:, class_idx]

                    # Filter by confidence threshold
                    mask = cls_scores > confidence_threshold
                    if not mask.any():
                        continue

                    scores_filtered = cls_scores[mask]
                    boxes_filtered = img_boxes[mask]
                    landmarks_filtered = img_landmarks[mask]
                    gaze_filtered = img_gaze[mask]

                    # Keep top-k candidates *before* NMS
                    if scores_filtered.size(0) > top_k:
                        scores_filtered, top_idx = scores_filtered.topk(top_k)
                        boxes_filtered = boxes_filtered[top_idx]
                        landmarks_filtered = landmarks_filtered[top_idx]
                        gaze_filtered = gaze_filtered[top_idx]

                    # Apply Non-Maximum Suppression (NMS) per class
                    keep_indices = torchvision.ops.nms(boxes_filtered, scores_filtered, nms_threshold)

                    final_boxes.append(boxes_filtered[keep_indices])
                    final_scores.append(scores_filtered[keep_indices])
                    final_labels.append(torch.full_like(scores_filtered[keep_indices], actual_class_label, dtype=torch.long))

                    # Append landmarks/gaze only if it's the Face class (label 1)
                    if actual_class_label == 1: # Assuming 1 is Face
                       final_landmarks.append(landmarks_filtered[keep_indices])
                       final_gaze.append(gaze_filtered[keep_indices])
                    # If it's another class (Phone), append placeholders or handle differently
                    # else:
                    #    # Append tensors of zeros with matching number of detections if needed
                    #    num_kept = len(keep_indices)
                    #    final_landmarks.append(torch.zeros(num_kept, img_landmarks.size(1), device=x.device))
                    #    final_gaze.append(torch.zeros(num_kept, img_gaze.size(1), device=x.device))


                # Combine results from all classes for the image
                if not final_boxes: # No detections for this image
                     results.append({
                        'boxes': torch.empty((0, 4), device=x.device),
                        'scores': torch.empty((0,), device=x.device),
                        'labels': torch.empty((0,), dtype=torch.long, device=x.device),
                        'landmarks': torch.empty((0, img_landmarks.size(1)), device=x.device),
                        'gaze': torch.empty((0, img_gaze.size(1)), device=x.device)
                    })
                else:
                    img_final_boxes = torch.cat(final_boxes, dim=0)
                    img_final_scores = torch.cat(final_scores, dim=0)
                    img_final_labels = torch.cat(final_labels, dim=0)
                    # Careful concatenation for landmarks/gaze if placeholders were used
                    img_final_landmarks = torch.cat(final_landmarks, dim=0) if final_landmarks else torch.empty((0, img_landmarks.size(1)), device=x.device)
                    img_final_gaze = torch.cat(final_gaze, dim=0) if final_gaze else torch.empty((0, img_gaze.size(1)), device=x.device)


                    results.append({
                        'boxes': img_final_boxes,
                        'scores': img_final_scores,
                        'labels': img_final_labels,
                        'landmarks': img_final_landmarks,
                        'gaze': img_final_gaze
                    })

            return results
