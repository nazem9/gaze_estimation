# utils.py
import torch
import torchvision

def cxcywh_to_xyxy(boxes_cxcywh):
    """ Convert boxes from (cx, cy, w, h) to (xmin, ymin, xmax, ymax) format. """
    xmin = boxes_cxcywh[..., 0] - boxes_cxcywh[..., 2] / 2
    ymin = boxes_cxcywh[..., 1] - boxes_cxcywh[..., 3] / 2
    xmax = boxes_cxcywh[..., 0] + boxes_cxcywh[..., 2] / 2
    ymax = boxes_cxcywh[..., 1] + boxes_cxcywh[..., 3] / 2
    return torch.stack((xmin, ymin, xmax, ymax), dim=-1)

def xyxy_to_cxcywh(boxes_xyxy):
    """ Convert boxes from (xmin, ymin, xmax, ymax) to (cx, cy, w, h) format. """
    cx = (boxes_xyxy[..., 0] + boxes_xyxy[..., 2]) / 2
    cy = (boxes_xyxy[..., 1] + boxes_xyxy[..., 3]) / 2
    w = boxes_xyxy[..., 2] - boxes_xyxy[..., 0]
    h = boxes_xyxy[..., 3] - boxes_xyxy[..., 1]
    return torch.stack((cx, cy, w, h), dim=-1)

def calculate_iou(boxes1_xyxy, boxes2_xyxy):
    """ Calculate IoU between two sets of boxes (xyxy format). """
    # Get intersection coordinates
    x_min_inter = torch.max(boxes1_xyxy[:, None, 0], boxes2_xyxy[None, :, 0])
    y_min_inter = torch.max(boxes1_xyxy[:, None, 1], boxes2_xyxy[None, :, 1])
    x_max_inter = torch.min(boxes1_xyxy[:, None, 2], boxes2_xyxy[None, :, 2])
    y_max_inter = torch.min(boxes1_xyxy[:, None, 3], boxes2_xyxy[None, :, 3])

    # Calculate intersection area
    inter_width = torch.clamp(x_max_inter - x_min_inter, min=0)
    inter_height = torch.clamp(y_max_inter - y_min_inter, min=0)
    intersection_area = inter_width * inter_height

    # Calculate individual box areas
    area1 = (boxes1_xyxy[:, 2] - boxes1_xyxy[:, 0]) * (boxes1_xyxy[:, 3] - boxes1_xyxy[:, 1])
    area2 = (boxes2_xyxy[:, 2] - boxes2_xyxy[:, 0]) * (boxes2_xyxy[:, 3] - boxes2_xyxy[:, 1])

    # Calculate union area
    union_area = area1[:, None] + area2[None, :] - intersection_area

    # Calculate IoU
    iou = intersection_area / torch.clamp(union_area, min=1e-6) # Avoid division by zero
    return iou # Shape: (num_boxes1, num_boxes2)

def encode_boxes(gt_boxes_xyxy, default_boxes_cxcywh, variance=(0.1, 0.2)):
    """
    Encode ground truth boxes relative to default boxes.
    Args:
        gt_boxes_xyxy (Tensor): Ground truth boxes (N, 4) in xyxy format.
        default_boxes_cxcywh (Tensor): Default boxes (N, 4) in cxcywh format.
        variance (tuple): Variance values for encoding.

    Returns:
        Tensor: Encoded box offsets (N, 4).
    """
    default_boxes_xyxy = cxcywh_to_xyxy(default_boxes_cxcywh)
    gt_boxes_cxcywh = xyxy_to_cxcywh(gt_boxes_xyxy)

    # Calculate offsets (SSD formula)
    g_cx = (gt_boxes_cxcywh[:, 0] - default_boxes_cxcywh[:, 0]) / default_boxes_cxcywh[:, 2]
    g_cy = (gt_boxes_cxcywh[:, 1] - default_boxes_cxcywh[:, 1]) / default_boxes_cxcywh[:, 3]
    g_w = torch.log(gt_boxes_cxcywh[:, 2] / default_boxes_cxcywh[:, 2])
    g_h = torch.log(gt_boxes_cxcywh[:, 3] / default_boxes_cxcywh[:, 3])

    # Apply variance
    encoded = torch.stack([g_cx, g_cy, g_w, g_h], dim=-1)
    encoded /= torch.tensor(variance, device=encoded.device).repeat(2) # Var applied to cx,cy and w,h
    return encoded

def decode_boxes(pred_offsets, default_boxes_cxcywh, variance=(0.1, 0.2)):
    """
    Decode predicted offsets back to absolute box coordinates.
    Args:
        pred_offsets (Tensor): Predicted box offsets (N, 4).
        default_boxes_cxcywh (Tensor): Default boxes (N, 4) in cxcywh format.
        variance (tuple): Variance values used during encoding.

    Returns:
        Tensor: Decoded boxes (N, 4) in xyxy format.
    """
    # Apply variance
    offsets = pred_offsets * torch.tensor(variance, device=pred_offsets.device).repeat(2)

    # Decode coordinates
    pred_cx = offsets[:, 0] * default_boxes_cxcywh[:, 2] + default_boxes_cxcywh[:, 0]
    pred_cy = offsets[:, 1] * default_boxes_cxcywh[:, 3] + default_boxes_cxcywh[:, 1]
    pred_w = torch.exp(offsets[:, 2]) * default_boxes_cxcywh[:, 2]
    pred_h = torch.exp(offsets[:, 3]) * default_boxes_cxcywh[:, 3]

    # Convert to xyxy
    pred_boxes_cxcywh = torch.stack([pred_cx, pred_cy, pred_w, pred_h], dim=-1)
    pred_boxes_xyxy = cxcywh_to_xyxy(pred_boxes_cxcywh)
    return pred_boxes_xyxy

# Basic NMS function (can use torchvision.ops.nms)
def non_max_suppression(boxes_xyxy, scores, iou_threshold):
    """ Performs Non-Maximum Suppression. """
    return torchvision.ops.nms(boxes_xyxy, scores, iou_threshold)

# TODO: Implement landmark encoding/decoding if needed (e.g., relative to anchor center)
# For simplicity, the loss might directly regress absolute landmark coords (normalized 0-1)
# or regress offsets from anchor center. If regressing offsets:

def encode_landmarks(gt_landmarks_norm, default_boxes_cxcywh, variance_lmk=(0.1,)):
    """ Encode landmarks relative to anchor box center and size. """
    # gt_landmarks_norm shape: (N_faces, num_landmarks, 2) -> (N_faces, 10) flattened
    # default_boxes_cxcywh shape: (N_faces, 4) - the anchors matched to faces
    num_landmarks = gt_landmarks_norm.shape[1] // 2
    gt_landmarks_flat = gt_landmarks_norm.view(-1, num_landmarks, 2) # (N_faces, 5, 2)

    # Anchor center (cx, cy) and size (w, h)
    anchor_cx = default_boxes_cxcywh[:, 0].unsqueeze(1) # (N_faces, 1)
    anchor_cy = default_boxes_cxcywh[:, 1].unsqueeze(1) # (N_faces, 1)
    anchor_w = default_boxes_cxcywh[:, 2].unsqueeze(1) # (N_faces, 1)
    anchor_h = default_boxes_cxcywh[:, 3].unsqueeze(1) # (N_faces, 1)

    # Calculate offsets: (gt_lm_x - anchor_cx) / anchor_w
    offset_x = (gt_landmarks_flat[..., 0] - anchor_cx) / anchor_w
    offset_y = (gt_landmarks_flat[..., 1] - anchor_cy) / anchor_h

    # Apply variance and flatten
    encoded = torch.stack([offset_x, offset_y], dim=-1) # (N_faces, 5, 2)
    encoded = encoded / torch.tensor(variance_lmk, device=encoded.device)
    return encoded.view(-1, num_landmarks * 2) # (N_faces, 10)


def decode_landmarks(pred_lmk_offsets, default_boxes_cxcywh, variance_lmk=(0.1,)):
    """ Decode landmark offsets back to normalized coordinates. """
    # pred_lmk_offsets shape: (N_faces, 10)
    # default_boxes_cxcywh shape: (N_faces, 4)
    num_landmarks = pred_lmk_offsets.shape[1] // 2
    pred_lmk_offsets_var = pred_lmk_offsets * torch.tensor(variance_lmk, device=pred_lmk_offsets.device)
    pred_lmk_offsets_reshaped = pred_lmk_offsets_var.view(-1, num_landmarks, 2) # (N_faces, 5, 2)

    # Anchor center (cx, cy) and size (w, h)
    anchor_cx = default_boxes_cxcywh[:, 0].unsqueeze(1) # (N_faces, 1)
    anchor_cy = default_boxes_cxcywh[:, 1].unsqueeze(1) # (N_faces, 1)
    anchor_w = default_boxes_cxcywh[:, 2].unsqueeze(1) # (N_faces, 1)
    anchor_h = default_boxes_cxcywh[:, 3].unsqueeze(1) # (N_faces, 1)

    # Decode: pred_lm_x = offset_x * anchor_w + anchor_cx
    decoded_x = pred_lmk_offsets_reshaped[..., 0] * anchor_w + anchor_cx
    decoded_y = pred_lmk_offsets_reshaped[..., 1] * anchor_h + anchor_cy

    decoded_landmarks = torch.stack([decoded_x, decoded_y], dim=-1) # (N_faces, 5, 2)
    return decoded_landmarks.view(-1, num_landmarks * 2) # (N_faces, 10)