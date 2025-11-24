import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU()
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = F.pad(
            x1,
            [
                diff_x // 2,
                diff_x - diff_x // 2,
                diff_y // 2,
                diff_y - diff_y // 2,
            ],
        )
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class LearnedUpsample(nn.Module):
    def __init__(self, in_channels, out_channels, final_size):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.adapt = nn.AdaptiveAvgPool2d(output_size=final_size)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.adapt(x)
        return x


class FiLM(nn.Module):
    def __init__(self, feature_dim, cond_dim, pool_kernel=2):
        super().__init__()
        self.pool_kernel = pool_kernel
        self.linear = nn.Linear(cond_dim, feature_dim * 2)

    def forward(self, x, loc_feat):
        loc_feat = loc_feat.permute(0, 2, 1)
        loc_feat = F.avg_pool1d(loc_feat, kernel_size=self.pool_kernel)
        loc_feat = loc_feat.permute(0, 2, 1)
        gamma_beta = self.linear(loc_feat)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        gamma = gamma.permute(0, 2, 1).unsqueeze(2)
        beta = beta.permute(0, 2, 1).unsqueeze(2)
        return gamma * x + beta


class CatConvFuse(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
        )

    def forward(self, x1, x2):
        return self.conv(torch.cat([x1, x2], dim=1))


class LocEncoder(nn.Module):
    def __init__(self, input_dim=4, output_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, loc_feat):
        return self.encoder(loc_feat)


class GatingFusion(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.attn_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(in_channels, 2, kernel_size=1),
        )

    def forward(self, feat_gnss, feat_sm):
        x_cat = torch.cat([feat_gnss, feat_sm], dim=1)
        attn_logits = self.attn_conv(x_cat)
        attn_weights = F.softmax(attn_logits, dim=1)
        gnss_weight = attn_weights[:, 0:1, :, :]
        sm_weight = attn_weights[:, 1:2, :, :]
        fused = gnss_weight * feat_gnss + sm_weight * feat_sm
        return fused


class SimpleFusion(nn.Module):
    def __init__(self, wave_feat_dim, tab_dim):
        super().__init__()
        self.tab_fusion_conv1 = nn.Conv2d(
            wave_feat_dim + tab_dim, wave_feat_dim, kernel_size=1
        )
        self.tab_fusion_bn1 = nn.BatchNorm2d(wave_feat_dim)
        self.tab_fusion_relu1 = nn.LeakyReLU()
        self.tab_fusion_conv2 = nn.Conv2d(
            wave_feat_dim, wave_feat_dim, kernel_size=3, padding=1
        )
        self.tab_fusion_bn2 = nn.BatchNorm2d(wave_feat_dim)
        self.tab_fusion_relu2 = nn.LeakyReLU()

    def forward(self, wave_feat, tab_feat):
        b, c, h, w = wave_feat.size()
        tab_feat_exp = (
            tab_feat.unsqueeze(-1).unsqueeze(-1).expand(-1, tab_feat.shape[1], h, w)
        )
        fused = torch.cat([wave_feat, tab_feat_exp], dim=1)
        fused = self.tab_fusion_conv1(fused)
        fused = self.tab_fusion_bn1(fused)
        fused = self.tab_fusion_relu1(fused)
        fused = self.tab_fusion_conv2(fused)
        fused = self.tab_fusion_bn2(fused)
        fused = self.tab_fusion_relu2(fused)
        return fused


class TabularMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.fc(x)


class FiLMEncoderFull(nn.Module):
    def __init__(self, n_channels=6, bilinear=True, loc_dim=4):
        super().__init__()
        self.inc = DoubleConv(n_channels, 16)
        self.down1 = Down(16, 32)
        self.down2 = Down(32, 64)
        self.down3 = Down(64, 128)
        factor = 2 if bilinear else 1
        self.down4 = Down(128, 256 // factor)
        self.loc_encoder = LocEncoder(input_dim=loc_dim, output_dim=64)
        self.film1 = FiLM(32, 64, pool_kernel=2)
        self.film2 = FiLM(64, 64, pool_kernel=4)
        self.film3 = FiLM(128, 64, pool_kernel=8)
        self.film4 = FiLM(256 // factor, 64, pool_kernel=16)

    def forward(self, x, loc_feat):
        loc_feat = self.loc_encoder(loc_feat)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x2 = self.film1(x2, loc_feat)
        x3 = self.down2(x2)
        x3 = self.film2(x3, loc_feat)
        x4 = self.down3(x3)
        x4 = self.film3(x4, loc_feat)
        x5 = self.down4(x4)
        x5 = self.film4(x5, loc_feat)
        return x5, x4, x3, x2, x1


class FiLMEncoderLast(FiLMEncoderFull):
    def forward(self, x, loc_feat):
        x5, x4, x3, x2, x1 = super().forward(x, loc_feat)
        return x5


class UNetDecoderCore(nn.Module):
    def __init__(self, in_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up1 = Up(in_channels + 128, 128, bilinear)
            self.up2 = Up(192, 64, bilinear)
            self.up3 = Up(96, 32, bilinear)
            self.up4 = Up(48, 16, bilinear)
        else:
            self.up1 = Up(in_channels, 128, bilinear)
            self.up2 = Up(128, 64, bilinear)
            self.up3 = Up(64, 32, bilinear)
            self.up4 = Up(32, 16, bilinear)

    def forward(self, x5, x4, x3, x2, x1):
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return x
    
class GlobalFeatureDecoder(nn.Module):
    def __init__(self, in_channels, hidden_dims, use_layernorm=None):
        super().__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        if use_layernorm is None:
            use_layernorm = [False] * len(hidden_dims)
        layers = []
        input_dim = in_channels * 2
        for dim, use_ln in zip(hidden_dims, use_layernorm):
            layers.append(nn.Linear(input_dim, dim))
            if use_ln:
                layers.append(nn.LayerNorm(dim))
            layers.append(nn.SiLU())
            input_dim = dim
        layers.append(nn.Linear(input_dim, 1))
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        feat_avg = self.global_avg_pool(x).view(x.size(0), -1)
        feat_max = self.global_max_pool(x).view(x.size(0), -1)
        features = torch.cat([feat_avg, feat_max], dim=1)
        out = self.fc(features)
        return out
