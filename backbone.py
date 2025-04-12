# backbone.py
import torch
import torch.nn as nn
import torchvision.models as models

class ResNetSSDBackbone(nn.Module):
    """
    ResNet-50 backbone for SSD, extracting features from intermediate
    layers and adding extra convolutional layers.
    """
    def __init__(self, pretrained=True):
        super().__init__()

        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)

        # --- Extract features from ResNet ---
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3 # Output channels: 1024, Spatial: /16
        self.layer4 = resnet.layer4 # Output channels: 2048, Spatial: /32

        # --- Extra layers for SSD ---
        self.extra_layers = nn.ModuleList([
            # Conv8_1, Conv8_2: Input 2048 -> Output 512, Spatial /64 (e.g., 5x5)
            nn.Sequential(
                nn.Conv2d(in_channels=2048, out_channels=256, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True)
            ),
            # Conv9_1, Conv9_2: Input 512 -> Output 256, Spatial /128 (e.g., 3x3)
            nn.Sequential(
                nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True)
            ),
            # Conv10_1, Conv10_2: Input 256 -> Output 256, Spatial /256 (e.g., 1x1)
            nn.Sequential(
                nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
                # This layer reduces to 1x1 if input is 3x3
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=0),
                nn.ReLU(inplace=True)
            ),
            # Conv11_1, Conv11_2: Input 256 (1x1) -> Output 256 (1x1)
             nn.Sequential(
                nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
                # ****************************************************** #
                # MODIFICATION HERE: Use kernel_size=1 for 1x1 input
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, stride=1, padding=0),
                # ****************************************************** #
                nn.ReLU(inplace=True)
            )
        ])

        # Initialize extra layers (important!)
        self._init_extra_layers()

    def _init_extra_layers(self):
        for layer_group in self.extra_layers:
            for layer in layer_group:
                if isinstance(layer, nn.Conv2d):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0.)

    def forward(self, x):
        """
        Passes input through the ResNet backbone and extra layers.

        Args:
            x (torch.Tensor): Input image tensor (B, C, H, W).

        Returns:
            List[torch.Tensor]: List of feature maps from selected layers.
                                (From layer3, layer4, and extra layers).
        """
        features = []

        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        features.append(x) # Feature map 1 (from layer3)

        x = self.layer4(x)
        features.append(x) # Feature map 2 (from layer4)

        # Pass through extra layers
        for layer in self.extra_layers:
            x = layer(x)
            features.append(x) # Feature maps 3, 4, 5, 6

        return features