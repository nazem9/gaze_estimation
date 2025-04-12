# -------------------------------------------------------------------
# ssd_model.py (with a matching anchor generator)
# This code references the config above so that the total anchors in the model
# will match the total anchors in DefaultBoxGenerator (2256).

import torch
import torch.nn as nn
import itertools
import math
from backbone import ResNetSSDBackbone
from config import NUM_CLASSES, NUM_LANDMARKS, FEATURE_MAP_CHANNELS, NUM_ANCHORS_PER_LOC
from config import ANCHOR_SIZES, ANCHOR_ASPECT_RATIOS, DEVICE, IMG_SIZE

class PredictionHead(nn.Module):
    """ A single prediction head for classification/regression. """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.conv.weight)
        if self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0.)

    def forward(self, x):
        return self.conv(x)


class DefaultBoxGenerator:
    """ Generates default boxes (anchors) in SSD style. """
    def __init__(self, feature_map_sizes, image_size, anchor_sizes, anchor_aspect_ratios):
        """
        Args:
            feature_map_sizes (list): e.g. [(19,19), (10,10), (5,5), (3,3), (1,1), (1,1)]
            image_size (int): input image size (assume square).
            anchor_sizes (list): e.g. [[45], [90,120], [150,180]...]
            anchor_aspect_ratios (list): matching aspect ratios for each layer
        """
        self.feature_map_sizes = feature_map_sizes
        self.image_size = float(image_size)
        self.anchor_sizes = anchor_sizes
        self.anchor_aspect_ratios = anchor_aspect_ratios
        assert len(feature_map_sizes) == len(anchor_sizes) == len(anchor_aspect_ratios), \
            "All must have same length (one per feature map)."

        self.num_levels = len(feature_map_sizes)
        self.default_boxes = self._generate_all_boxes()

    def _generate_all_boxes(self):
        all_boxes = []
        for k in range(self.num_levels):
            fm_h, fm_w = self.feature_map_sizes[k]
            sizes = self.anchor_sizes[k]
            ratios = self.anchor_aspect_ratios[k]

            for i, j in itertools.product(range(fm_h), range(fm_w)):
                # center coords, normalized
                cx = (j + 0.5) / fm_w
                cy = (i + 0.5) / fm_h

                for size in sizes:
                    # Always generate an anchor at ratio=1
                    s_k = size / self.image_size
                    all_boxes.append([cx, cy, s_k, s_k])

                    # For each ratio != 1.0, generate a box
                    for ar in ratios:
                        # skip if ar is 1.0, because we already did ratio=1 above
                        if abs(ar - 1.0) < 1e-6:
                            continue
                        ar_sqrt = math.sqrt(ar)
                        w = s_k * ar_sqrt
                        h = s_k / ar_sqrt
                        all_boxes.append([cx, cy, w, h])

        default_boxes_tensor = torch.tensor(all_boxes, dtype=torch.float32)
        default_boxes_tensor.clamp_(min=0.0, max=1.0)  # ensure in [0,1]
        return default_boxes_tensor  # shape (num_total_anchors, 4)

    def get_boxes(self):
        return self.default_boxes.to(DEVICE)


class CombinedSSD(nn.Module):
    """
    SSD model: ResNet-50 backbone + classification, bbox, landmark, and gaze heads.
    """
    def __init__(self, num_classes, num_landmarks, backbone_channels, num_anchors_list):
        super().__init__()

        self.num_classes = num_classes
        self.num_landmarks = num_landmarks
        self.backbone_channels = backbone_channels
        self.num_anchors_list = num_anchors_list

        self.backbone = ResNetSSDBackbone(pretrained=True)

        # Prediction heads
        self.cls_heads = nn.ModuleList()
        self.box_heads = nn.ModuleList()
        self.lmk_heads = nn.ModuleList()
        self.gaze_heads = nn.ModuleList()

        for i, in_c in enumerate(self.backbone_channels):
            num_anchors = self.num_anchors_list[i]
            self.cls_heads.append(PredictionHead(in_c, num_anchors * (self.num_classes + 1)))
            self.box_heads.append(PredictionHead(in_c, num_anchors * 4))
            self.lmk_heads.append(PredictionHead(in_c, num_anchors * self.num_landmarks * 2))
            self.gaze_heads.append(PredictionHead(in_c, num_anchors * 2))

    def forward(self, x):
        """
        Returns (B, num_total_anchors, num_classes+1),
                (B, num_total_anchors, 4),
                (B, num_total_anchors, num_landmarks*2),
                (B, num_total_anchors, 2)
        """
        features = self.backbone(x)
        batch_size = x.size(0)

        cls_outs, box_outs, lmk_outs, gaze_outs = [], [], [], []

        for i, feat in enumerate(features):
            # Each head
            cls_logits = self.cls_heads[i](feat)
            box_preds = self.box_heads[i](feat)
            lmk_preds = self.lmk_heads[i](feat)
            gaze_preds = self.gaze_heads[i](feat)

            # Permute and reshape
            cls_logits = cls_logits.permute(0,2,3,1).contiguous()
            box_preds  = box_preds.permute(0,2,3,1).contiguous()
            lmk_preds  = lmk_preds.permute(0,2,3,1).contiguous()
            gaze_preds = gaze_preds.permute(0,2,3,1).contiguous()

            cls_logits = cls_logits.view(batch_size, -1, self.num_classes + 1)
            box_preds  = box_preds.view(batch_size, -1, 4)
            lmk_preds  = lmk_preds.view(batch_size, -1, self.num_landmarks*2)
            gaze_preds = gaze_preds.view(batch_size, -1, 2)

            cls_outs.append(cls_logits)
            box_outs.append(box_preds)
            lmk_outs.append(lmk_preds)
            gaze_outs.append(gaze_preds)

        final_cls_logits = torch.cat(cls_outs, dim=1)    # (B, sum(all_anchors), C+1)
        final_box_preds  = torch.cat(box_outs, dim=1)    # (B, sum(all_anchors), 4)
        final_lmk_preds  = torch.cat(lmk_outs, dim=1)    # (B, sum(all_anchors), 10)
        final_gaze_preds = torch.cat(gaze_outs, dim=1)   # (B, sum(all_anchors), 2)

        return final_cls_logits, final_box_preds, final_lmk_preds, final_gaze_preds


def create_combined_ssd(num_classes=NUM_CLASSES, num_landmarks=NUM_LANDMARKS):
    model = CombinedSSD(
        num_classes=num_classes,
        num_landmarks=num_landmarks,
        backbone_channels=FEATURE_MAP_CHANNELS,
        num_anchors_list=NUM_ANCHORS_PER_LOC
    )
    return model


def create_default_box_generator(img_size=IMG_SIZE):
    # The exact feature map sizes from the backbone:
    #   layer0 => 19×19
    #   layer1 => 10×10
    #   layer2 => 5×5
    #   layer3 => 3×3
    #   layer4 => 1×1
    #   layer5 => 1×1
    feature_map_dims = [(19,19), (10,10), (5,5), (3,3), (1,1), (1,1)]

    generator = DefaultBoxGenerator(
        feature_map_sizes=feature_map_dims,
        image_size=img_size,
        anchor_sizes=ANCHOR_SIZES,
        anchor_aspect_ratios=ANCHOR_ASPECT_RATIOS
    )
    return generator


if __name__ == '__main__':
    model = create_combined_ssd().to(DEVICE)
    print(model)

    dummy_input = torch.randn(4, 3, 300, 300).to(DEVICE)
    with torch.no_grad():
        cls_logits, box_preds, lmk_preds, gaze_preds = model(dummy_input)

    print("\nOutput Shapes:")
    print("Class Logits:   ", cls_logits.shape)   
    print("Box Preds:      ", box_preds.shape)     
    print("Landmark Preds: ", lmk_preds.shape)    
    print("Gaze Preds:     ", gaze_preds.shape)

    generator = create_default_box_generator(img_size=300)
    default_boxes = generator.get_boxes()
    print("\nDefault Boxes Shape:", default_boxes.shape)

    num_anchors_calculated = default_boxes.shape[0]
    print("Total Anchors Generated:", num_anchors_calculated)

    # Let's confirm they match:
    print("Pred anchor count from model:", cls_logits.shape[1])
    if cls_logits.shape[1] == num_anchors_calculated:
        print("All good! Anchors match between model and generator.")
    else:
        print("Mismatch! Adjust anchor config or the model's anchors.")