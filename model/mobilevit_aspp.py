import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# Atrous Spatial Pyramid Pooling (ASPP)

class ASPP(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.atrous_block1 = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, dilation=1)
        self.atrous_block6 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=18, dilation=18)
        self.image_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.image_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1)
        self.conv_1x1_output = nn.Conv2d(out_ch * 5, out_ch, kernel_size=1, stride=1)

    def forward(self, x):
        size = x.shape[2:]
        image_features = self.image_pool(x)
        image_features = self.image_conv(image_features)
        image_features = F.interpolate(image_features, size=size, mode='bilinear', align_corners=False)

        out1 = self.atrous_block1(x)
        out2 = self.atrous_block6(x)
        out3 = self.atrous_block12(x)
        out4 = self.atrous_block18(x)

        out = torch.cat([out1, out2, out3, out4, image_features], dim=1)
        out = self.conv_1x1_output(out)
        return out

# Decoder Block

class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch + skip_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


# MobileViT_ASPP Model

class MobileViT_ASPP(nn.Module):
    def __init__(self, model_name="mobilevit_s", num_classes=1):
        super().__init__()
        # Create pretrained backbone
        self.backbone = timm.create_model(model_name, pretrained=True, features_only=True)
        # Use last four hierarchical feature maps for decoder
        chs = self.backbone.feature_info.channels()[-4:]  # f1,f2,f3,f4
        # ASPP on deepest feature map
        self.aspp = ASPP(chs[-1], 256)
        # Decoder stages with skip connections
        self.dec3 = DecoderBlock(256, chs[2], 128)
        self.dec2 = DecoderBlock(128, chs[1], 64)
        self.dec1 = DecoderBlock(64, chs[0], 32)
        # Final 1x1 convolution
        self.segmentation_head = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        H, W = x.shape[2:]
        feats = self.backbone(x)
        f1, f2, f3, f4 = feats[-4:]

        x = self.aspp(f4)
        x = self.dec3(x, f3)
        x = self.dec2(x, f2)
        x = self.dec1(x, f1)

        seg_logits = self.segmentation_head(x)
        seg_logits = F.interpolate(seg_logits, size=(H, W), mode='bilinear', align_corners=False)
        return seg_logits