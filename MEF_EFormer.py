import torch
import torch.nn as nn
import torch.nn.functional as F


class MDTA(nn.Module):
    def __init__(self, channels, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.qkv_conv = nn.Conv2d(
            channels * 3, channels * 3,
            kernel_size=3, padding=1,
            groups=channels * 3, bias=False
        )
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv_conv(self.qkv(x)).chunk(3, dim=1)
        q = q.reshape(b, self.num_heads, -1, h * w)
        k = k.reshape(b, self.num_heads, -1, h * w)
        v = v.reshape(b, self.num_heads, -1, h * w)
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1).contiguous()) * self.temperature, dim=-1)
        out = self.project_out(torch.matmul(attn, v).reshape(b, -1, h, w))
        return out

class GDFN(nn.Module):
    def __init__(self, channels, expansion_factor):
        super().__init__()
        hidden_channels = int(channels * expansion_factor)
        self.project_in = nn.Conv2d(channels, hidden_channels * 2, kernel_size=1, bias=False)
        self.conv = nn.Conv2d(
            hidden_channels * 2, hidden_channels * 2,
            kernel_size=3, padding=1,
            groups=hidden_channels * 2, bias=False
        )
        self.project_out = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        x1, x2 = self.conv(self.project_in(x)).chunk(2, dim=1)
        x = self.project_out(F.gelu(x1) * x2)
        return x

class RestormerBlock(nn.Module):
    def __init__(self, channels, num_heads, expansion_factor=2.66):
        super().__init__()
        
        self.channels = int(channels)
        self.num_heads = int(num_heads)
        self.expansion_factor = float(expansion_factor)

        self.norm1 = nn.LayerNorm(self.channels)
        self.attn = MDTA(self.channels, self.num_heads)
        self.norm2 = nn.LayerNorm(self.channels)
        self.ffn = GDFN(self.channels, self.expansion_factor)

        self.head_dim = self.channels // self.num_heads

    def forward(self, x):
        b, c, h, w = x.shape
        x_ln1 = self.norm1(x.reshape(b, c, -1).transpose(-2, -1).contiguous()) \
                    .transpose(-2, -1).contiguous().reshape(b, c, h, w)
        x = x + self.attn(x_ln1)

        x_ln2 = self.norm2(x.reshape(b, c, -1).transpose(-2, -1).contiguous()) \
                    .transpose(-2, -1).contiguous().reshape(b, c, h, w)
        x = x + self.ffn(x_ln2)
        return x



class CAFB(nn.Module):   # LLFormer
    def __init__(self, in_dim,bias=True):
        super(CAFB, self).__init__()
        self.chanel_in = in_dim

        self.temperature = nn.Parameter(torch.ones(1))

        self.qkv = nn.Conv2d( self.chanel_in ,  self.chanel_in *3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(self.chanel_in*3, self.chanel_in*3, kernel_size=3, stride=1, padding=1, groups=self.chanel_in*3, bias=bias)
        self.project_out = nn.Conv2d(self.chanel_in, self.chanel_in, kernel_size=1, bias=bias)

    def forward(self,x):
        m_batchsize, N, C, height, width = x.size()
        x_input = x.view(m_batchsize,N*C, height, width)
        qkv = self.qkv_dwconv(self.qkv(x_input))
        q, k, v = qkv.chunk(3, dim=1)
        q = q.view(m_batchsize, N, -1)
        k = k.view(m_batchsize, N, -1)
        v = v.view(m_batchsize, N, -1)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out_1 = (attn @ v)
        out_1 = out_1.view(m_batchsize, -1, height, width)

        out_1 = self.project_out(out_1)
        out_1 = out_1.view(m_batchsize, N, C, height, width)

        out = out_1+x
        out = out.view(m_batchsize, -1, height, width)
        return out



class SSE_Restormer(nn.Module):
    def __init__(self, num_blocks=(1, 1, 1, 1), num_heads=(1, 2, 2, 4), channels=(64, 64, 64, 64)):
        super().__init__()
        expansion_factor = 2.66
        self.encoders = nn.ModuleList([
            nn.Sequential(*[
                RestormerBlock(ch, hd, expansion_factor)
                for _ in range(nb)
            ])
            for nb, hd, ch in zip(num_blocks, num_heads, channels)
        ])

    def forward(self, f0):
        x = f0
        feats = []

        for stage in self.encoders:
            x = stage(x)
            feats.append(x)

        return tuple(feats)



class CAM(nn.Module):
    """
    Branch-wise channel attention fusion:
    F1 -> CAM -> sigmoid -> w1
    F2 -> CAM -> sigmoid -> w2
    returns w1, w2 where each branch has its own channel weights.
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(4, channels // reduction)
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, F1, F2):
        if F2.shape[2:] != F1.shape[2:]:
            F2 = F.interpolate(F2, size=F1.shape[2:], mode="bilinear", align_corners=False)

        B, C, H, W = F1.size()
        y1 = self.global_pool(F1).flatten(1)
        y2 = self.global_pool(F2).flatten(1)

        w1 = self.fc1(y1).view(B, C, 1, 1)
        w2 = self.fc2(y2).view(B, C, 1, 1)

        return w1, w2

class SAFM(nn.Module):
    def __init__(self, channels, kernel_size=7):
        super(SAFM, self).__init__()
        padding = 3

        self.conv = nn.Conv2d(4, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, F1, F2):
        avg1_out = torch.mean(F1, dim=1, keepdim=True)
        max1_out, _ = torch.max(F1, dim=1, keepdim=True)
        avg2_out = torch.mean(F2, dim=1, keepdim=True)
        max2_out, _ = torch.max(F2, dim=1, keepdim=True)
        
        x_cat = torch.cat([avg1_out, max1_out, avg2_out, max2_out], dim=1) #Concatenate before the conv
        x_out = self.conv(x_cat) #Pass to conv layer

        sigma = self.sigmoid(x_out)
        F1_attended = sigma * F1
        F2_attended = (1 - sigma) * F2
        
        return  F1_attended + F2_attended

class GFFM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.ca = CAM(channels)
        self.ss = SAFM(channels, kernel_size=7)

    def forward(self, F1, F2):
        if F2.shape[2:] != F1.shape[2:]:
            F2 = F.interpolate(F2, size=F1.shape[2:], mode="bilinear", align_corners=False)

        w1, w2 = self.ca(F1, F2)
        return self.ss(w1*F1, w2*F2)


class MEF_EFormer(nn.Module):
    def __init__(self):
        super().__init__()
        self.channels   = [64,64,64,64]
        self.expansion_factor   = 2.66
        self.num_refine_blocks = 2

        self.num_layers =4
        C = 64
        N = 4

        # ---- stems ----
        self.stem_u = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.stem_o = nn.Conv2d(3, 64, 3, 1, 1, bias=False)

        # ---- encoders ----
        self.Ulayers = SSE_Restormer(
            num_blocks=[1,1,1,1],
            num_heads=[1,2,2,4],
            channels=[64,64,64,64],
        )

        self.Olayers = SSE_Restormer(
            num_blocks=[1,1,1,1],
            num_heads=[1,2,2,4],
            channels=[64,64,64,64],
        )

        # ---- Fused Layers ----
        self.FuseLayers = nn.ModuleList([
            GFFM(channels=64)
            for i in range(self.num_layers)
        ])

        # ---- CAFB ----
        self.dec = CAFB(in_dim=N * C) 
        self.dec_reduce = nn.Conv2d(N * C, C, kernel_size=1, bias=True)
        
        self.refine_layer = nn.Sequential(*[
            RestormerBlock(
                channels=64,
                num_heads=1,
                expansion_factor=2.66,
            )
            for _ in range(2)
        ])

        self.proj = nn.Conv2d(self.channels[-1], 3, kernel_size=1, stride=1, padding=0, bias=True)



    def forward(self, under, over):
        u0 = self.stem_u(under)
        o0 = self.stem_o(over)

        Ufeats = self.Ulayers(u0)  # e.g., (u1,u2,u3,u4)
        Ofeats = self.Olayers(o0)  # e.g., (o1,o2,o3,o4)

        fused_list = []

        for i in range(self.num_layers):
            fused = self.FuseLayers[i](Ufeats[i], Ofeats[i])  # (2C -> C_i)
            fused_list.append(fused)

        F_5d = torch.stack(fused_list, dim=1)  # (B, N, C, H, W)
        F = self.dec(F_5d)                     # (B, N*C, H, W)
        F = self.dec_reduce(F)                 # (B, N*C, H, W) -> (B, C, H, W)
   
        refined = self.refine_layer(F)
        F = F + refined

        output = self.proj(F)

        return output

    @torch.no_grad()
    def inference(self, under, over):
        self.eval()
        out = self.forward(under, over)
        out = torch.clamp(out, 0.0, 1.0)
        return out