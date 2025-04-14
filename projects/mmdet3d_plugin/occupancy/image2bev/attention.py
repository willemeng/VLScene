

from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
import gc
import time
from einops import rearrange, reduce
from torch import nn, einsum
from timm.models.layers import DropPath

class DisparityRegression(nn.Module):
    def __init__(self, maxdisp):
        super(DisparityRegression, self).__init__()
        self.maxdisp = maxdisp

    def forward(self, x):
        assert(x.is_contiguous() == True)
        with torch.cuda.device_of(x):
            disp = torch.reshape(torch.arange(0, self.maxdisp, device=torch.cuda.current_device(), dtype=torch.float32),[1,self.maxdisp,1,1])
            disp = disp.repeat(x.size()[0], 1, x.size()[2], x.size()[3])
            out = torch.sum(x * disp, 1)
        return out

class Disp(nn.Module):
    def __init__(self, maxdisp=192):
        super(Disp, self).__init__()
        self.maxdisp = maxdisp
        self.softmax = nn.Softmin(dim=1)
        self.disparity = DisparityRegression(maxdisp=self.maxdisp)

    def forward(self, x):
        x = F.interpolate(x, [self.maxdisp, x.size()[3]*3, x.size()[4]*3], mode='trilinear', align_corners=False)
        x = torch.squeeze(x, 1)
        x = self.softmax(x)      
        x = self.disparity(x)
        return x


        
class attention(nn.Module):
 
    def __init__(self, in_dim):
        super(attention, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim  , kernel_size=1)
        self.key_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim , kernel_size=1)
        self.value_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))   

        self.softmax = nn.Softmax(dim=-1)   

    def forward(self, q, kv):
 
        x=kv
        m_batchsize, C, D, height, width = x.size()

        confidence  = F.softmax(q, dim=2)   
        confidence = torch.max(confidence, dim=2)[0]  
        confidence =confidence.view(m_batchsize, -1, width * height) 

     
        proj_query = self.query_conv(q).view(m_batchsize, -1, width * height).permute(0, 2, 1)
 
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
 
        energy = torch.bmm(proj_query, proj_key)
 
        attention = self.softmax(energy)

        attention = confidence*attention 

 
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)
   
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))   
     
        out = out.view(m_batchsize, C, D, height, width)

        out = self.gamma * out + x
        return out



class CA3D(nn.Module):
    def __init__(self, channel):
        super(CA3D, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(channel, channel, kernel_size=3, stride=1, dilation=1, padding=1),
            nn.GELU(),
            nn.GroupNorm(1, channel),
            )
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.conv2 = nn.Sequential(
            nn.Conv3d(channel, channel//8, kernel_size=1, stride=1, dilation=1, padding=0),
            nn.GELU(),
            nn.Conv3d(channel//8, channel, kernel_size=1, stride=1, dilation=1, padding=0),
            nn.GELU(),
  
        )

        self.sigmoid = nn.Sigmoid()
        self.conv = nn.Sequential(
            nn.Conv3d(channel, channel, kernel_size=3, stride=1, dilation=1, padding=1, groups=1),
            nn.GELU(),
            nn.GroupNorm(1, channel),
        )
    def forward(self, x):
        data = self.conv1(x)
        pool = self.avg_pool(data)
        squeeze = self.conv2(pool)
        weight = self.sigmoid(squeeze)
        out = weight*data
        out = self.conv(out)
        return out

class CrossAttentionFusion(nn.Module):
    def __init__(self, voxel_dim=128, text_dim=512, hidden_dim=128, n_heads=4):
        super(CrossAttentionFusion, self).__init__()
        self.voxel_dim = voxel_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        
        # Define the multihead attention layer
        self.multihead_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=n_heads)

        # Linear layers to match dimensions for attention
        self.voxel_proj = nn.Linear(voxel_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        
        # Output layer
        self.out_proj = nn.Linear(hidden_dim, voxel_dim)

    def forward(self, voxel_features, text_features):
        # Reshape voxel features to (seq_len, batch_size, feature_dim)
        B, C, X, Y, Z = voxel_features.shape
        voxel_features = voxel_features.reshape(B, C, X*Y*Z).permute(2, 0, 1)  # (D*H*W, B, C)
        text_features = text_features.unsqueeze(1).repeat(1, B, 1)  # (seq_len, batch_size, channels)

        # Reshape text features to (seq_len, batch_size, feature_dim)
        text_features = text_features.to(voxel_features.dtype)  # (seq_len, batch_size, feature_dim)
        
        # Project voxel and text features to the same hidden dimension
        voxel_features = self.voxel_proj(voxel_features)
        text_features = self.text_proj(text_features)
        
        # Apply cross-attention
        attn_output, attn_weights = self.multihead_attn(voxel_features, text_features, text_features)

        # Reshape the output back to the original voxel shape
        # attn_output = attn_output
        fused_features = self.out_proj(attn_output).permute(1, 2, 0).view(B, C, X, Y, Z)
        
        return fused_features


class Attention_Cross(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, q):
        # kv = self.to_kv(x).chunk(2, dim = -1)
        # k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), kv)
        B, H, W, C = x.shape
        N = H * W
        kv = self.to_kv(x).reshape(B, H, W, 2, self.heads, self.dim_head).permute(3, 0, 4, 1, 2, 5)
        k, v = kv[0], kv[1]
        q = q.reshape(B, H, W, self.heads, self.dim_head).permute(0, 3, 1, 2, 4)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 3, 1, 4).reshape(B, H, W, C)
        #out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        # qkv = self.to_qkv(x).chunk(3, dim = -1)
        # q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        B, H, W, C = x.shape
        N = H * W
        qkv = self.to_qkv(x).reshape(B, H, W, 3, self.heads, self.dim_head).permute(3, 0, 4, 1, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        #out = rearrange(out, 'b h n d -> b n (h d)')
        out = out.permute(0, 2, 3, 1, 4).reshape(B, H, W, C)
        return self.to_out(out)

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


class CrossLayer(nn.Module):
    def __init__(self, dim, num_heads, kernel_size=7,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, layer_scale=None):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        ##### global attention
        self.attn = Attention_Cross(dim, heads = num_heads, dim_head =(dim//num_heads), dropout=attn_drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.layer_scale = False
        if layer_scale is not None and type(layer_scale) in [int, float]:
            self.layer_scale = True
            self.gamma1 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)

    def forward(self, x, q_extra):
        x=x.permute(0,2,3,1)
        q_extra=q_extra.permute(0,2,3,1)
        if not self.layer_scale:
            shortcut = x
            x = self.norm1(x)
            x = self.attn(x, q_extra)
            x = shortcut + self.drop_path(x)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            x = x.permute(0,3,1,2)
            return x
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x, q_extra)
        x = shortcut + self.drop_path(self.gamma1 * x)
        x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
        x = x.permute(0,3,1,2)
        # q_extra=q_extra.permute(0,3,1,2)
        return x
    
class AttentionLayer(nn.Module):
    def __init__(self, dim, num_heads, kernel_size=7,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, layer_scale=None):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)

        self.attn = Attention(dim, heads = num_heads, dim_head =(dim//num_heads), dropout=attn_drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.layer_scale = False
        if layer_scale is not None and type(layer_scale) in [int, float]:
            self.layer_scale = True
            self.gamma1 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)

    def forward(self, x):
        x=x.permute(0,2,3,1)
        if not self.layer_scale:
            shortcut = x
            x = self.norm1(x)
            x = self.attn(x)
            x = shortcut + self.drop_path(x)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            x=x.permute(0,3,1,2)
            return x
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = shortcut + self.drop_path(self.gamma1 * x)
        x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
        return x

class ConvDownsampler(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.reduction = nn.Conv2d(dim, 2 * dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.norm = norm_layer(2 * dim)

    def forward(self, x):
        x = self.reduction(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x = self.norm(x)
        return x



class ClipFusion(nn.Module):
    def __init__(self, channel):
        super().__init__()


        self.attention_sem = nn.Sequential(
             nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel, kernel_size=1),
            nn.Sigmoid()
        )

        self.attention_img = nn.Sequential(
             nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel, kernel_size=1),
            nn.Sigmoid()
        )
        self.adapter_sem = nn.Conv2d(channel, channel, 1)
        self.adapter_img = nn.Conv2d(channel, channel, 1)

    def forward(self, img_features, sem_features):
        sem_features = self.adapter_sem(sem_features)
        img_features = self.adapter_img(img_features)

        attn_sem = self.attention_sem(sem_features)
        attn_img = self.attention_img(img_features)

        fusion_features = torch.mul(sem_features, attn_sem) \
            + torch.mul(img_features, attn_img)

        return fusion_features


if __name__ == '__main__':
    cross_layer = ClipFusion(channel=128).cuda()
    image_f = torch.randn([1, 128, 48, 160]).cuda()
    clip_f = torch.randn([1, 128, 48, 160]).cuda()
    x_out = cross_layer(clip_f, image_f)
    print(x_out.shape)

