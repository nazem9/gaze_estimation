import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import calculate_iou, encode_boxes, encode_landmarks, cxcywh_to_xyxy
from config import DEVICE, IOU_THRESHOLD_MATCHING, NEG_POS_RATIO, ALPHA_BBOX_LOSS, ALPHA_LANDMARK_LOSS, ALPHA_GAZE_LOSS, CLASS_MAP, NUM_LANDMARKS

class SSDLoss(nn.Module):
    def __init__(self, default_boxes_cxcywh, iou_threshold=IOU_THRESHOLD_MATCHING,
                 neg_pos_ratio=NEG_POS_RATIO, alpha_bbox=ALPHA_BBOX_LOSS,
                 alpha_lmk=ALPHA_LANDMARK_LOSS, alpha_gaze=ALPHA_GAZE_LOSS):
        super().__init__()
        self.default_boxes = default_boxes_cxcywh.to(DEVICE)
        self.default_boxes_xyxy = cxcywh_to_xyxy(self.default_boxes)
        self.iou_threshold = iou_threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.alpha_bbox = alpha_bbox
        self.alpha_lmk = alpha_lmk
        self.alpha_gaze = alpha_gaze
        self.face_class_idx = CLASS_MAP['face']
        self.smooth_l1_loss = nn.SmoothL1Loss(reduction='none')
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, predictions, targets):
        cls_logits, box_preds, lmk_preds, gaze_preds = predictions
        batch_size = cls_logits.size(0)
        num_anchors = self.default_boxes.size(0)
        num_classes = cls_logits.size(2) - 1

        all_gt_classes = []
        all_gt_box_offsets = []
        all_gt_lmk_offsets = []
        all_gt_gaze = []

        for i in range(batch_size):
            gt_boxes = targets[i]['boxes'].to(DEVICE)
            gt_labels = targets[i]['labels'].to(DEVICE)
            gt_landmarks = targets[i].get('landmarks')
            gt_gaze_targets = targets[i].get('gaze')

            if gt_landmarks is not None:
                gt_landmarks = gt_landmarks.to(DEVICE)
            if gt_gaze_targets is not None:
                gt_gaze_targets = gt_gaze_targets.to(DEVICE)

            num_gt_objects = gt_boxes.size(0)
            gt_classes_img = torch.zeros(num_anchors, dtype=torch.long, device=DEVICE)
            gt_box_offsets_img = torch.zeros((num_anchors, 4), dtype=torch.float32, device=DEVICE)
            gt_lmk_offsets_img = torch.zeros((num_anchors, NUM_LANDMARKS * 2), dtype=torch.float32, device=DEVICE)
            gt_gaze_img = torch.zeros((num_anchors, 2), dtype=torch.float32, device=DEVICE)

            if num_gt_objects == 0:
                all_gt_classes.append(gt_classes_img)
                all_gt_box_offsets.append(gt_box_offsets_img)
                all_gt_lmk_offsets.append(gt_lmk_offsets_img)
                all_gt_gaze.append(gt_gaze_img)
                continue

            iou = calculate_iou(gt_boxes, self.default_boxes_xyxy)

            best_anchor_iou, best_anchor_idx = iou.max(dim=1)
            best_gt_iou, best_gt_idx = iou.max(dim=0)

            anchor_matches_gt_idx = torch.full((num_anchors,), -1, dtype=torch.long, device=DEVICE)
            anchor_matches_gt_idx[best_anchor_idx] = torch.arange(num_gt_objects, device=DEVICE)
            match_mask = best_gt_iou >= self.iou_threshold
            anchor_matches_gt_idx[match_mask] = best_gt_idx[match_mask]

            positive_anchors_mask = anchor_matches_gt_idx >= 0
            matched_gt_indices = anchor_matches_gt_idx[positive_anchors_mask]

            gt_classes_img[positive_anchors_mask] = gt_labels[matched_gt_indices]
            matched_gt_boxes = gt_boxes[matched_gt_indices]
            matched_default_boxes = self.default_boxes[positive_anchors_mask]
            gt_box_offsets_img[positive_anchors_mask] = encode_boxes(matched_gt_boxes, matched_default_boxes)

            is_face_anchor = (gt_classes_img == self.face_class_idx) & positive_anchors_mask
            if gt_landmarks is not None and is_face_anchor.any():
                face_anchor_gt_indices = anchor_matches_gt_idx[is_face_anchor]
                face_obj_mask_in_gt = (gt_labels == self.face_class_idx)
                original_face_indices = torch.where(face_obj_mask_in_gt)[0]
                map_gt_idx_to_face_idx = {orig_idx.item(): face_tensor_idx for face_tensor_idx, orig_idx in enumerate(original_face_indices)}
                indices_in_face_tensors = torch.tensor([map_gt_idx_to_face_idx[idx.item()] for idx in face_anchor_gt_indices if idx.item() in map_gt_idx_to_face_idx], dtype=torch.long, device=DEVICE)

                if indices_in_face_tensors.numel() > 0:
                    matched_gt_landmarks = gt_landmarks[indices_in_face_tensors]
                    matched_face_default_boxes = self.default_boxes[is_face_anchor]
                    gt_lmk_offsets_img[is_face_anchor] = encode_landmarks(matched_gt_landmarks, matched_face_default_boxes)

                    if gt_gaze_targets is not None:
                        matched_gt_gaze = gt_gaze_targets[indices_in_face_tensors]
                        gt_gaze_img[is_face_anchor] = matched_gt_gaze

            all_gt_classes.append(gt_classes_img)
            all_gt_box_offsets.append(gt_box_offsets_img)
            all_gt_lmk_offsets.append(gt_lmk_offsets_img)
            all_gt_gaze.append(gt_gaze_img)

        gt_classes_batch = torch.stack(all_gt_classes)
        gt_box_offsets_batch = torch.stack(all_gt_box_offsets)
        gt_lmk_offsets_batch = torch.stack(all_gt_lmk_offsets)
        gt_gaze_batch = torch.stack(all_gt_gaze)

        positive_mask = gt_classes_batch > 0
        num_positives = positive_mask.sum(dim=1, keepdim=True).clamp(min=1.0)

        cls_loss_all = self.cross_entropy_loss(cls_logits.view(-1, num_classes + 1), gt_classes_batch.view(-1)).view(batch_size, num_anchors)
        cls_loss_pos = cls_loss_all[positive_mask].sum()

        cls_loss_neg_hard = torch.zeros(batch_size, device=DEVICE)
        for i in range(batch_size):
            neg_mask = ~positive_mask[i]
            num_pos = int(num_positives[i].item())
            num_neg = min(int(neg_mask.sum().item()), self.neg_pos_ratio * num_pos)

            if num_neg > 0:
                neg_losses = cls_loss_all[i][neg_mask]
                topk_losses, _ = torch.topk(neg_losses, num_neg)
                cls_loss_neg_hard[i] = topk_losses.sum()

        total_cls_loss = (cls_loss_pos + cls_loss_neg_hard.sum()) / num_positives.sum()

        box_loss_all = self.smooth_l1_loss(box_preds, gt_box_offsets_batch)
        total_box_loss = box_loss_all[positive_mask].sum() / num_positives.sum()

        face_anchor_mask = (gt_classes_batch == self.face_class_idx) & positive_mask
        num_face_positives = face_anchor_mask.sum(dim=1, keepdim=True).clamp(min=1.0)

        lmk_loss_all = self.smooth_l1_loss(lmk_preds, gt_lmk_offsets_batch)
        total_lmk_loss = lmk_loss_all[face_anchor_mask].sum() / num_face_positives.sum()

        gaze_loss_all = self.smooth_l1_loss(gaze_preds, gt_gaze_batch)
        total_gaze_loss = gaze_loss_all[face_anchor_mask].sum() / num_face_positives.sum()

        total_loss = total_cls_loss + self.alpha_bbox * total_box_loss + self.alpha_lmk * total_lmk_loss + self.alpha_gaze * total_gaze_loss

        return total_loss, total_cls_loss, total_box_loss, total_lmk_loss, total_gaze_loss
