import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTaskLoss(nn.Module):
    """
    Loss function for face detection, landmark localization, and gaze estimation
    """
    def __init__(self, lambda_face=1.0, lambda_landmark=1.0, lambda_gaze=1.0):
        super(MultiTaskLoss, self).__init__()
        self.lambda_face = lambda_face
        self.lambda_landmark = lambda_landmark
        self.lambda_gaze = lambda_gaze
        
    def forward(self, predictions, targets):
        """
        Compute the combined loss
        
        Args:
            predictions: Dict with 'face_preds', 'landmark_preds', 'gaze_preds'
            targets: Dict with 'boxes', 'landmarks', 'gaze', 'matches', 'labels'
        """
        face_preds = predictions['face_preds']
        landmark_preds = predictions['landmark_preds']
        gaze_preds = predictions['gaze_preds']
        
        # Get matches (which anchor boxes match which ground truth)
        matches = targets['matches']  # Shape: [batch_size, num_gt_boxes]
        labels = targets['labels']    # Shape: [batch_size, num_anchors]
        
        # Face detection loss (classification + regression)
        face_cls_loss = F.binary_cross_entropy_with_logits(
            face_preds[:, :, 4], 
            labels,
            reduction='sum'
        )
        
        # Box regression loss (only for positive matches)
        pos_mask = labels > 0
        if pos_mask.sum() > 0:
            # Get target boxes
            target_boxes = targets['boxes']  # Shape: [batch_size, num_gt_boxes, 4]
            
            # Map ground truth boxes to anchors based on matches
            batch_size = face_preds.size(0)
            box_targets = []
            
            for i in range(batch_size):
                # Get matched boxes for this image
                matched_boxes = target_boxes[i][matches[i]]
                box_targets.append(matched_boxes)
            
            box_targets = torch.stack(box_targets, dim=0)
            
            # Compute smooth L1 loss for box regression
            box_loss = F.smooth_l1_loss(
                face_preds[:, :, :4][pos_mask],
                box_targets[pos_mask],
                reduction='sum'
            )
        else:
            box_loss = torch.tensor(0.0).to(face_preds.device)
        
        face_loss = face_cls_loss + box_loss
        
        # Landmark regression loss (only for positive matches)
        if pos_mask.sum() > 0:
            # Get target landmarks
            target_landmarks = targets['landmarks']  # Shape: [batch_size, num_gt_boxes, 136]
            
            # Map ground truth landmarks to anchors based on matches
            batch_size = landmark_preds.size(0)
            landmark_targets = []
            
            for i in range(batch_size):
                # Get matched landmarks for this image
                matched_landmarks = target_landmarks[i][matches[i]]
                landmark_targets.append(matched_landmarks)
            
            landmark_targets = torch.stack(landmark_targets, dim=0)
            
            # Compute smooth L1 loss for landmark regression
            landmark_loss = F.smooth_l1_loss(
                landmark_preds[pos_mask],
                landmark_targets[pos_mask],
                reduction='sum'
            )
        else:
            landmark_loss = torch.tensor(0.0).to(landmark_preds.device)
        
        # Gaze regression loss (only for positive matches)
        if pos_mask.sum() > 0:
            # Get target gaze
            target_gaze = targets['gaze']  # Shape: [batch_size, num_gt_boxes, 2]
            
            # Map ground truth gaze to anchors based on matches
            batch_size = gaze_preds.size(0)
            gaze_targets = []
            
            for i in range(batch_size):
                # Get matched gaze for this image
                matched_gaze = target_gaze[i][matches[i]]
                gaze_targets.append(matched_gaze)
            
            gaze_targets = torch.stack(gaze_targets, dim=0)
            
            # Compute smooth L1 loss for gaze regression
            gaze_loss = F.smooth_l1_loss(
                gaze_preds[pos_mask],
                gaze_targets[pos_mask],
                reduction='sum'
            )
        else:
            gaze_loss = torch.tensor(0.0).to(gaze_preds.device)
        
        # Compute the total loss
        total_loss = (
            self.lambda_face * face_loss + 
            self.lambda_landmark * landmark_loss + 
            self.lambda_gaze * gaze_loss
        )
        
        return {
            'total_loss': total_loss,
            'face_loss': face_loss,
            'landmark_loss': landmark_loss,
            'gaze_loss': gaze_loss
        }
