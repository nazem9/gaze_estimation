import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model Configuration
NUM_CLASSES = 2  # face, phone (background is implicit class 0 in loss)
CLASS_MAP = {"face": 1, "phone": 2}
LABEL_MAP = {v: k for k, v in CLASS_MAP.items()}
LABEL_MAP[0] = 'background'

NUM_LANDMARKS = 5  # Number of facial landmarks

# ResNet50 Backbone feature maps
FEATURE_MAP_CHANNELS = [1024, 2048, 512, 256, 256, 256]

# -------------------------------------------------------------------
# Revised anchor config to match NUM_ANCHORS_PER_LOC exactly

# For each layer, #anchors = (#sizes) × (#aspect_ratios)
# so that it matches NUM_ANCHORS_PER_LOC = [4, 6, 6, 6, 4, 4].

ANCHOR_SIZES = [
    [45],         # Layer0: 1 size => with 4 aspect-ratios => 4 anchors
    [90, 120],    # Layer1: 2 sizes => 3 aspect-ratios => 6 anchors total
    [150, 180],   # Layer2
    [210, 240],   # Layer3
    [270],        # Layer4
    [320],        # Layer5
]
ANCHOR_ASPECT_RATIOS = [
    [1.0, 2.0, 0.5, 3.0],  # 1 size × 4 AR = 4 anchors for layer0
    [1.0, 2.0, 0.5],       # 2 sizes × 3 AR = 6 anchors for layer1
    [1.0, 2.0, 0.5],       # likewise 6 anchors for layer2
    [1.0, 2.0, 0.5],       # likewise 6 anchors for layer3
    [1.0, 2.0, 0.5, 3.0],  # 1 size × 4 AR = 4 anchors for layer4
    [1.0, 2.0, 0.5, 3.0],  # 1 size × 4 AR = 4 anchors for layer5
]

NUM_ANCHORS_PER_LOC = [4, 6, 6, 6, 4, 4]

# -------------------------------------------------------------------

# Training Configuration
IMG_SIZE = 300
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
NUM_EPOCHS = 100
WEIGHT_DECAY = 5e-4
MOMENTUM = 0.9

# Loss Configuration
IOU_THRESHOLD_MATCHING = 0.5
NEG_POS_RATIO = 3
ALPHA_BBOX_LOSS = 1.0
ALPHA_LANDMARK_LOSS = 1.0
ALPHA_GAZE_LOSS = 0.5

# Data paths
ANNOTATION_DIR = "generated_annotations_ssd/train"
IMAGE_BASE_DIR = "dataset/train/images"

# Post-processing
CONFIDENCE_THRESHOLD = 0.5
NMS_IOU_THRESHOLD = 0.45
MAX_DETECTIONS_PER_IMG = 200