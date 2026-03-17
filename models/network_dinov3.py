# -----------------------------------------------------------------------------------
# SwinIR: Image Restoration Using Swin Transformer, https://arxiv.org/abs/2108.10257
# Originally Written by Ze Liu, Modified by Jingyun Liang.
# -----------------------------------------------------------------------------------

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import trunc_normal_
from einops import rearrange
import numbers
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '6'
GPU_number = os.environ['CUDA_VISIBLE_DEVICES']

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

##########################################################################
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)
    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(
            dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(
            hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.kv = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        b, c, h, w = x.shape
        q = self.q_dwconv(self.q(x))
        kv = self.kv_dwconv(self.kv(y))
        k, v = kv.chunk(2, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class CrossTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(CrossTransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.norm1_ = LayerNorm(dim, LayerNorm_type)
        self.attn = CrossAttention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x, y):
        x = x + self.attn(self.norm1(x), self.norm1_(y))
        x = x + self.ffn(self.norm2(x))
        return x

class Conv_BN_ReLU(nn.Module):
    def __init__(self, dim):
        super(Conv_BN_ReLU, self).__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size=1,padding=0,stride=1)
        self.bn = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, feature):
        feature = self.conv(feature)
        feature = self.bn(feature)
        feature = self.relu(feature)
        return feature


class ConvBlock(nn.Module):
    """A basic convolutional block with Conv -> BN -> ReLU."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))

class UpConv(nn.Module):
    """An upsampling module with Upsample -> ConvBlock."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = ConvBlock(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.up(x))

class DinoV3_UNet(nn.Module):
    """
    A U-Net-like decoder architecture using a DINOv3 Vision Transformer as the encoder.
    """
    def __init__(self, backbone, dim):
        super().__init__()

        # --- Encoder ---
        self.backbone = backbone
        self.backbone_embed_dim = self.backbone.embed_dim
        self.skip_connection_layers = [2,5,8,11]

        print(f"Extracting features from layers: {self.skip_connection_layers}")

        # --- Decoder ---
        embed_dim = self.backbone_embed_dim
        self.upconv1 = UpConv(embed_dim, embed_dim // 2)
        self.upconv2 = UpConv(embed_dim, embed_dim // 2)
        self.upconv3 = UpConv(embed_dim, embed_dim // 2)

        self.upconv4 = UpConv(embed_dim // 2, embed_dim // 4)
        self.upconv5 = UpConv(embed_dim // 2, embed_dim // 4)

        self.upconv6 = UpConv(embed_dim // 4, embed_dim // 8)
        self.upconv7 = UpConv(embed_dim // 8, embed_dim // 16)

        # Skip connection from input
        self.input_skip_conv = ConvBlock(3, embed_dim // 16, kernel_size=3, stride=1, padding=1)
        self.final_conv = nn.Conv2d(embed_dim // 16 * 2, dim, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        img_h, img_w = x.shape[-2:]
        
        # --- Encoder Forward Pass ---
        # Get intermediate features from the backbone
        intermediate_outputs = self.backbone.get_intermediate_layers(
            x, 
            n=self.skip_connection_layers, 
            reshape=False, # Keep as (B, N, C) for the head
            return_class_token=False,
        )

        # Reshape features to [B, C, H, W]
        reshaped_features = []
        patch_h, patch_w = img_h // self.backbone.patch_size, img_w // self.backbone.patch_size
        for feat in intermediate_outputs:
            reshaped_features.append(feat.permute(0, 2, 1).reshape(x.shape[0], -1, patch_h, patch_w))
    
        f2, f5, f8, f11 = reshaped_features

        # --- Decoder ---
        # First level fusion
        f8_11 = self.upconv1(f8 + f11)
        f5_up = self.upconv2(f5)
        f2_up = self.upconv3(f2)

        # Second level fusion
        f5_8_11 = self.upconv4(f8_11 + f5_up)
        f2_up = self.upconv5(f2_up)

        # Third level fusion
        f2_5_8_11 = self.upconv6(f5_8_11 + f2_up)

        # Final upsampling
        out = self.upconv7(f2_5_8_11)
        
        # Skip connection from input
        input_skip = self.input_skip_conv(x)
        
        # Concat with input skip connection
        out = torch.cat([out, input_skip], dim=1)
        
        # Final prediction
        out = self.final_conv(out)
        # out = F.interpolate(out, size=(img_h, img_w), mode='bilinear', align_corners=False)

        return out

class Reconstruction(nn.Module):
    def __init__(self, embed_dim):
        super(Reconstruction, self).__init__()
        self.restormer1 = TransformerBlock(dim=embed_dim, num_heads=8, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias')
        self.restormer2 = TransformerBlock(dim=embed_dim, num_heads=8, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias')
        self.conv = nn.Conv2d(embed_dim, 3, 1, 1, 0)

    def forward(self, x):
        out = self.restormer1(x)
        out = self.restormer2(out)
        out = self.conv(out)
        return out

class DinoFusion(nn.Module):
    def __init__(self, img_size=64, patch_size=1, in_chans=1,
                 embed_dim=96, Ex_depths=[4], Fusion_depths=[2, 2], Re_depths=[4], 
                 Ex_num_heads=[6], Fusion_num_heads=[6, 6], Re_num_heads=[6],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, upscale=2, img_range=1., upsampler='', extractor=None, resi_connection='1conv',
                 **kwargs):
        super(DinoFusion, self).__init__()
        num_out_ch = in_chans
        num_feat = 64
        self.img_range = img_range
        embed_dim_temp = int(embed_dim / 2)
        self.window_size = 16
        print('in_chans: ', in_chans)
        if in_chans == 3 or in_chans == 6:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            rgbrgb_mean = (0.4488, 0.4371, 0.4040, 0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
            self.mean_in = torch.Tensor(rgbrgb_mean).view(1, 6, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)

        ################################### 1, shallow feature extraction ###################################
        self.extractor = extractor
        self.dino_proessor_A = DinoV3_UNet(extractor, embed_dim)
        self.dino_proessor_B = DinoV3_UNet(extractor, embed_dim)
        # self.head_A = Dino_head(dim=384)
        # self.head_B = Dino_head(dim=384)
        ################################### 2, deep feature extraction ######################################
        # self.conv_A = nn.Conv2d(384, embed_dim, 1,1,0)
        # self.conv_B = nn.Conv2d(384, embed_dim, 1,1,0)
        self.crossformer_A0 = CrossTransformerBlock(dim=embed_dim, num_heads=8, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias')
        self.crossformer_A1 = CrossTransformerBlock(dim=embed_dim, num_heads=8, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias')
        self.crossformer_A2 = CrossTransformerBlock(dim=embed_dim, num_heads=8, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias')
        self.crossformer_A3 = CrossTransformerBlock(dim=embed_dim, num_heads=8, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias')
        self.crossformer_B0 = CrossTransformerBlock(dim=embed_dim, num_heads=8, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias')
        self.crossformer_B1 = CrossTransformerBlock(dim=embed_dim, num_heads=8, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias')
        self.crossformer_B2 = CrossTransformerBlock(dim=embed_dim, num_heads=8, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias')
        self.crossformer_B3 = CrossTransformerBlock(dim=embed_dim, num_heads=8, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias')

        self.recA = Reconstruction(embed_dim)
        self.recB = Reconstruction(embed_dim)

        ################################ 3, high quality image reconstruction ################################
        self.conv_last1 = nn.Conv2d(embed_dim*2, embed_dim, 3, 1, 1)
        self.former_last = TransformerBlock(dim=embed_dim, num_heads=8, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias')
        self.conv_last2 = nn.Conv2d(embed_dim, int(embed_dim//2), 3, 1, 1)
        self.conv_last3 = nn.Conv2d(int(embed_dim/2), num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)
        self.lock_backbone()

    def lock_backbone(self):
        for p in self.extractor.parameters():
            p.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def dino_ectractor(self, x):
        features = self.extractor.get_intermediate_layers(
            x, n = [2, 5, 8, 11]
        )
        return features
    
    def process(self, input):
        if input.shape[1] == 1:
            input = torch.concat([input,input,input], 1)
        return input
    def forward(self, A, B):
        A = self.process(A)
        B = self.process(B)
        # print("Initializing the model")
        x = A
        y = B
        H, W = x.shape[2:]
        x = self.check_image_size(x)
        y = self.check_image_size(y)

        self.mean_A = self.mean.type_as(x)
        self.mean_B = self.mean.type_as(y)
        self.mean = (self.mean_A + self.mean_B) / 2

        x = (x - self.mean_A) * self.img_range
        y = (y - self.mean_B) * self.img_range

        # Feedforward
        # patch_h, patch_w = x.shape[-2] // 16, x.shape[-1] // 16
        # x = self.head_A(self.dino_ectractor(x), patch_h, patch_w)
        # y = self.head_B(self.dino_ectractor(y), patch_h, patch_w)

        # x = self.conv_A(x)
        # y = self.conv_B(y)

        x = self.dino_proessor_A(x)
        y = self.dino_proessor_B(y)
        x_res = x
        y_res = y
        x_ = self.recA(x)
        y_ = self.recB(y)

        x = self.crossformer_A0(x,y)
        y = self.crossformer_B0(y,x)
        x = self.crossformer_A1(x,y)
        y = self.crossformer_B1(y,x)
        x = self.crossformer_A2(x,y)
        y = self.crossformer_B2(y,x)
        x = self.crossformer_A3(x,y)
        y = self.crossformer_B3(y,x)

        x = self.conv_last1(torch.concat([x,y],1))
        x = self.former_last(x)
        x = self.conv_last2(x)
        x = self.conv_last3(x)
        
        x = x / self.img_range + self.mean
        # x_res = x_res/ self.img_range + self.mean
        # y_res = y_res/ self.img_range + self.mean
        return x[:, :, :H, :W], x_[:, :, :H, :W], y_[:, :, :H, :W]
        return x[:, :, :H, :W], x_[:, :, :H, :W], y_[:, :, :H, :W], x_res[:, :, :H, :W], y_res[:, :, :H, :W]
        return x


if __name__ == '__main__':
    upscale = 4
    window_size = 8
    height = 512
    width = 512
    
    repo_dir = r'/mnt/sdb/dusongcheng/data/code/dinov3/segdino-main/dinov3'
    dino_ckpt = '/mnt/sdb/dusongcheng/data/code/dinov3/pth/dinov3_vits16_pretrain_lvd1689m-08c60483.pth'
    dinov3_block = torch.hub.load(repo_dir, 'dinov3_vits16', source='local', weights=dino_ckpt).cuda()
    
    model = DinoFusion(upscale=2, img_size=(height, width), in_chans=3,
                   window_size=window_size, img_range=1., depths=[6, 6, 6, 6],
                   embed_dim=64, num_heads=[6, 6, 6, 6], mlp_ratio=2, upsampler='', extractor=dinov3_block).cuda()
    # print(model)
    # print(height, width, model.flops() / 1e9)

    x = torch.randn((1, 3, height, width)).cuda()
    y = torch.randn((1, 3, height, width)).cuda()
    _,_,x,_,_ = model(x,y)
    print('Parameters number of model is ', sum(param.numel() for param in model.parameters()))
    print(y.shape)
    print(x.shape)

