import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class DecoderBlock(nn.Module):
    """
    U-Net Decoder block with skip connection, upsampling, and double convolution.
    """
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super(DecoderBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor = None) -> torch.Tensor:
        # Bilinear upsample to match skip spatial dimensions
        if skip is not None:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
            x = torch.cat([x, skip], dim=1)
        else:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SiameseResNetUNet(nn.Module):
    """
    Siamese ResNet Backbone (ResNet18 / ResNet34) with Multi-Scale Feature Difference
    and U-Net Decoder for Bi-temporal Satellite Change Detection.

    Args:
        backbone (str): 'resnet18' or 'resnet34'
        pretrained (bool): Load ImageNet pre-trained weights for encoder
        fusion_mode (str): 'diff' (absolute difference) or 'concat_diff' (concat + absolute difference)
    """
    def __init__(self, backbone: str = 'resnet34', pretrained: bool = True, fusion_mode: str = 'concat_diff'):
        super(SiameseResNetUNet, self).__init__()
        self.fusion_mode = fusion_mode

        if backbone == 'resnet18':
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            encoder = models.resnet18(weights=weights)
            channels = [64, 64, 128, 256, 512]
        elif backbone == 'resnet34':
            weights = models.ResNet34_Weights.DEFAULT if pretrained else None
            encoder = models.resnet34(weights=weights)
            channels = [64, 64, 128, 256, 512]
        else:
            raise ValueError(f"Unsupported backbone: {backbone}. Choose 'resnet18' or 'resnet34'.")

        # Encoder stages (Shared weights for Siamese branches)
        self.layer0 = nn.Sequential(
            encoder.conv1,
            encoder.bn1,
            encoder.relu
        )
        self.maxpool = encoder.maxpool
        self.layer1 = encoder.layer1
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4

        # Fusion factor multiplier
        # 'diff' -> 1x channels, 'concat_diff' -> 3x channels (A, B, |A-B|)
        factor = 3 if fusion_mode == 'concat_diff' else 1

        # Multi-scale fusion projectors
        self.fuse4 = nn.Conv2d(channels[4] * factor, channels[4], kernel_size=1)
        self.fuse3 = nn.Conv2d(channels[3] * factor, channels[3], kernel_size=1)
        self.fuse2 = nn.Conv2d(channels[2] * factor, channels[2], kernel_size=1)
        self.fuse1 = nn.Conv2d(channels[1] * factor, channels[1], kernel_size=1)
        self.fuse0 = nn.Conv2d(channels[0] * factor, channels[0], kernel_size=1)

        # Decoder Blocks
        self.dec4 = DecoderBlock(channels[4], channels[3], 256)
        self.dec3 = DecoderBlock(256, channels[2], 128)
        self.dec2 = DecoderBlock(128, channels[1], 64)
        self.dec1 = DecoderBlock(64, channels[0], 32)
        self.dec0 = DecoderBlock(32, 0, 16)

        # Final Classification Head
        self.head = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=1)
        )

    def _fuse(self, f1: torch.Tensor, f2: torch.Tensor, projector: nn.Module) -> torch.Tensor:
        diff = torch.abs(f1 - f2)
        if self.fusion_mode == 'concat_diff':
            fused = torch.cat([f1, f2, diff], dim=1)
        else:
            fused = diff
        return projector(fused)

    def forward_encoder(self, x: torch.Tensor):
        c0 = self.layer0(x)         # (B, 64, H/2, W/2)
        p0 = self.maxpool(c0)       # (B, 64, H/4, W/4)
        c1 = self.layer1(p0)        # (B, 64, H/4, W/4)
        c2 = self.layer2(c1)        # (B, 128, H/8, W/8)
        c3 = self.layer3(c2)        # (B, 256, H/16, W/16)
        c4 = self.layer4(c3)        # (B, 512, H/32, W/32)
        return c0, c1, c2, c3, c4

    def forward(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for bi-temporal image pair.
        t1: Pre-temporal image (B, 3, H, W)
        t2: Post-temporal image (B, 3, H, W)
        Returns: Change probability logit (B, 1, H, W)
        """
        input_size = t1.shape[2:]

        # Siamese feature extraction
        f1_0, f1_1, f1_2, f1_3, f1_4 = self.forward_encoder(t1)
        f2_0, f2_1, f2_2, f2_3, f2_4 = self.forward_encoder(t2)

        # Multi-scale feature fusion
        f_0 = self._fuse(f1_0, f2_0, self.fuse0)
        f_1 = self._fuse(f1_1, f2_1, self.fuse1)
        f_2 = self._fuse(f1_2, f2_2, self.fuse2)
        f_3 = self._fuse(f1_3, f2_3, self.fuse3)
        f_4 = self._fuse(f1_4, f2_4, self.fuse4)

        # U-Net Decoding
        d4 = self.dec4(f_4, f_3)
        d3 = self.dec3(d4, f_2)
        d2 = self.dec2(d3, f_1)
        d1 = self.dec1(d2, f_0)
        d0 = self.dec0(d1)

        # Output logits
        out = self.head(d0)
        if out.shape[2:] != input_size:
            out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=True)

        return out


def create_model(backbone: str = 'resnet34', pretrained: bool = True) -> SiameseResNetUNet:
    """Helper function to instantiate model."""
    return SiameseResNetUNet(backbone=backbone, pretrained=pretrained)


if __name__ == '__main__':
    # Test model shape consistency
    dummy_t1 = torch.randn(2, 3, 256, 256)
    dummy_t2 = torch.randn(2, 3, 256, 256)
    model = create_model(backbone='resnet18', pretrained=False)
    preds = model(dummy_t1, dummy_t2)
    print(f"Model output shape: {preds.shape}")
    assert preds.shape == (2, 1, 256, 256), "Shape mismatch!"
    print("Model test passed successfully!")
