import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        variance, mean = torch.var_mean(x,dim=1,correction=0,keepdim=True,)
        x = (x - mean) * torch.rsqrt(variance + self.eps)
        return (x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1))



class MDTA(nn.Module):
    def __init__(self, channels, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.qkv_conv = nn.Conv2d(channels * 3,channels * 3,kernel_size=3,padding=1,groups=channels * 3,bias=False,)
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape

        q, k, v = self.qkv_conv(self.qkv(x)).chunk(3, dim=1)

        q = q.reshape(b, self.num_heads, -1, h * w)
        k = k.reshape(b, self.num_heads, -1, h * w)
        v = v.reshape(b, self.num_heads, -1, h * w)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = torch.matmul(q, k.transpose(-2, -1))
        attn = torch.softmax(attn * self.temperature, dim=-1)

        out = torch.matmul(attn, v)
        out = out.reshape(b, c, h, w)

        return self.project_out(out)

class GDFN(nn.Module):
    def __init__(self, channels, expansion_factor):
        super().__init__()
        hidden_channels = int(channels * expansion_factor)
        self.project_in = nn.Conv2d(channels, hidden_channels * 2, kernel_size=1, bias=False)
        self.conv = nn.Conv2d(hidden_channels * 2, hidden_channels * 2,kernel_size=3, padding=1,groups=hidden_channels * 2, bias=False)
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

        self.norm1 = LayerNorm2d(self.channels)
        self.attn = MDTA(self.channels, self.num_heads)
        self.norm2 = LayerNorm2d(self.channels)
        self.ffn = GDFN(self.channels, self.expansion_factor)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class CAFB(nn.Module):
    def __init__(self, in_dim, bias=True):
        super().__init__()
        self.chanel_in = in_dim
        self.temperature = nn.Parameter(torch.ones(1))
        self.qkv = nn.Conv2d(in_dim, in_dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(in_dim * 3, in_dim * 3, kernel_size=3, stride=1, padding=1, groups=in_dim * 3, bias=bias)
        self.project_out = nn.Conv2d(in_dim, in_dim, kernel_size=1, bias=bias)

    def _qkv_branch(self, x, start, end):
        pointwise_bias = self.qkv.bias[start:end] if self.qkv.bias is not None else None
        branch = F.conv2d(x, self.qkv.weight[start:end], pointwise_bias, stride=self.qkv.stride, padding=self.qkv.padding, dilation=self.qkv.dilation)

        depthwise_bias = self.qkv_dwconv.bias[start:end] if self.qkv_dwconv.bias is not None else None
        branch = F.conv2d(branch, self.qkv_dwconv.weight[start:end], depthwise_bias, stride=self.qkv_dwconv.stride, padding=self.qkv_dwconv.padding, dilation=self.qkv_dwconv.dilation, groups=end - start)

        return branch

    def forward(self, x):
        b, n, c, h, w = x.shape
        residual = x.reshape(b, n * c, h, w)
        branch_size = self.chanel_in

        q = self._qkv_branch(residual, 0, branch_size)
        q = F.normalize(q.reshape(b, n, -1), dim=-1)

        attention_columns = []

        for group in range(n):
            start = branch_size + group * c
            end = start + c
            k_group = self._qkv_branch(residual, start, end)
            k_group = F.normalize(k_group.reshape(b, 1, -1), dim=-1)
            attention_columns.append(torch.bmm(q, k_group.transpose(1, 2)))
            del k_group

        attention = torch.cat(attention_columns, dim=-1)
        attention = torch.softmax(attention * self.temperature, dim=-1)

        del q, attention_columns

        v = self._qkv_branch(residual, 2 * branch_size, 3 * branch_size)
        v = v.reshape(b, n, -1)

        out = torch.bmm(attention, v).reshape(b, n * c, h, w)

        del v, attention

        out = self.project_out(out)

        return out + residual


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
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(4, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, F1, F2):
        stats = torch.cat(
            (F1.mean(1, keepdim=True), F1.amax(1, keepdim=True),
             F2.mean(1, keepdim=True), F2.amax(1, keepdim=True)),
            dim=1,
        )
        sigma = torch.sigmoid(self.conv(stats)).to(F1.dtype)
        # sigma * F1 + (1 - sigma) * F2  ==  lerp(F2, F1, sigma) 
        return torch.lerp(F2, F1, sigma)

class GFFM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.ca = CAM(channels)
        self.ss = SAFM(kernel_size=7)

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

        C = self.channels[-1]
        N = self.num_layers

        # ---- stems ----
        self.stem_u = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.stem_o = nn.Conv2d(3, 64, 3, 1, 1, bias=False)

        # ---- encoders ----
        self.Ulayers = SSE_Restormer(num_blocks=[1,1,1,1],num_heads=[1,2,2,4],channels=[64,64,64,64],)

        self.Olayers = SSE_Restormer(num_blocks=[1,1,1,1],num_heads=[1,2,2,4],channels=[64,64,64,64],)

        # ---- Fused Layers ----
        self.FuseLayers = nn.ModuleList([GFFM(channels=64) for i in range(self.num_layers)])

        # ---- CAFB ----
        self.dec = CAFB(in_dim=N * C) 
        self.dec_reduce = nn.Conv2d(N * C, C, kernel_size=1, bias=True)
        
        self.refine_layer = nn.Sequential(*[
            RestormerBlock(channels=64,num_heads=1,expansion_factor=2.66,)
            for _ in range(self.num_refine_blocks)
        ])
        self.proj = nn.Conv2d(self.channels[-1], 3, kernel_size=1, stride=1, padding=0, bias=True)
        
    def forward(self, under, over):
        under = self.stem_u(under)
        over = self.stem_o(over)

        fused_list = []

        for under_stage, over_stage, fusion_stage in zip(self.Ulayers.encoders, self.Olayers.encoders, self.FuseLayers):
            under = under_stage(under)
            over = over_stage(over)
            fused_list.append(fusion_stage(under, over))

        del under, over

        fused = torch.stack(fused_list, dim=1)

        del fused_list

        fused = self.dec(fused)
        fused = self.dec_reduce(fused)

        fused = fused + self.refine_layer(fused)
      

        return self.proj(fused)
    
    @torch.no_grad()
    def inference(self, under, over):
        self.eval()
        out = self.forward(under, over)
        out = torch.clamp(out, 0.0, 1.0)
        return out