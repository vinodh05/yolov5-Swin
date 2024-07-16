import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from functools import lru_cache, reduce
from operator import mul
from models.trunc_nomal_timm import trunc_normal_

def get_window_size(x_size, window_size, shift_size = None):
    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0
    if shift_size is None:
        return tuple(use_window_size)
    return tuple(use_window_size), tuple(use_shift_size)

def window_partition(x,window_size):
    B,D,H,W,C = x.shape
    x = x.view(B,D//window_size[0], window_size[0], H//window_size[1], window_size[1], W//window_size[2],window_size[2],C)
    windows = x.permute(0,1,3,5,2,4,6,7).contiguous().view(-1, reduce(mul, window_size), C)
    return windows

def window_reverse(windows, window_size, B, D, H, W):
    """
    Args:
        windows: (B*num_windows, window_size, window_size, C)
        window_size (tuple[int]): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, D, H, W, C)
    """
    x = windows.view(B, D // window_size[0], H // window_size[1], W // window_size[2], window_size[0], window_size[1], window_size[2], -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)
    return x

def drop_path(x, drop_prob: float = 0, training: bool = False, scale_by_keep: bool = True):
    if drop_prob == 0 or not training:
        return x
    keep_prob = 1- drop_prob
    shape = (x.shape[0],) +(1,)*(x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x*random_tensor

class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0, scale_by_keep: bool = True):
        super(DropPath,self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep
        
    def forward(self,x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)
    def extra_repr(self):
        return f'drop_prob = {round(self.drop_prob,3):0.3f}'

class WindowAttention3D(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias = False, qk_scale = None, attn_drop = 0, proj_drop = 0):
        super().__init__()
        self.dim =  dim
        self.window_Size = window_size
        self.num_heads = num_heads
        head_dim = dim //num_heads 
        self.scale = qk_scale  or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2*window_size[0] -1)*(2*window_size[1]-1)*(2*window_size[2] -1), num_heads)
        )
        coords_d = torch.arange(self.window_Size[0])
        coords_h = torch.arange(self.window_Size[1])
        coords_w = torch.arange(self.window_Size[2])
        coords = torch.stack(torch.meshgrid(coords_d,coords_h,coords_w, indexing = 'ij'))
        coords_flatten = torch.flatten(coords,1)
        relative_coords = coords_flatten[:,:,None] - coords_flatten[:,None,:]
        relative_coords = relative_coords.permute(1,2,0).contiguous()
        relative_coords[:,:,0] += self.window_Size[0]-1
        relative_coords[:,:,1] += self.window_Size[1]-1
        relative_coords[:,:,2] += self.window_Size[2]-1

        relative_coords[:,:,0] *= (2*self.window_Size[1] - 1) * (2*self.window_Size[2] - 1)
        relative_coords[:,:,1] *= (2*self.window_Size[2] -1)
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        
        self.qkv = nn.Linear(dim, dim*3, bias = qkv_bias)
        self.attn_drop =  nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim,dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std = 0.02)
        self.softmax = nn.Softmax(dim = -1)

    def forward(self, x, mask = None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_,N,3,self.num_heads, C//self.num_heads).permute(2,0,3,1,4)
        q,k,v = qkv[0],qkv[1],qkv[2]

        q = q*self.scale
        attn = q @ k.transpose(-2,-1)
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index[:N,:N].reshape(-1)].reshape(N,N,-1)
        relative_position_bias = relative_position_bias.permute(2,0,1).contiguous()
        attn=attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ //nW,nW, self.num_heads, N,N)+mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1,self.num_heads,N,N)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1,2).reshape(B_,N,C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x    

@lru_cache()
def create_mask(D, H, W, window_size, shift_size, device, dtype_model=torch.float32):
    D = int(np.ceil(D / window_size[0])) * window_size[0]
    H = int(np.ceil(H / window_size[1])) * window_size[1]
    W = int(np.ceil(W / window_size[2])) * window_size[2]
    
    img_mask = torch.zeros((1, D, H, W, 1), dtype=dtype_model, device=device)  # 1 Dp Hp Wp 1
    cnt = 0
    for d in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0],None):
        for h in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1],None):
            for w in slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2],None):
                img_mask[:, d, h, w, :] = cnt
                cnt += 1
    mask_windows = window_partition(img_mask, window_size)  # nW, ws[0]*ws[1]*ws[2], 1
    mask_windows = mask_windows.squeeze(-1)  # nW, ws[0]*ws[1]*ws[2]
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features = None, out_features = None, act_layer = nn.GELU, drop = 0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self,x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class SwinTransformerLayer3D(nn.Module):
    def __init__(self,dim,num_heads,window_size = (2,7,7),shift_size = (0,0,0),mlp_ratio = 4,qkv_bias = False,qk_scale = None,
                 drop = 0, attn_drop = 0,drop_path = 0,act_layer = nn.GELU, norm_layer = nn.LayerNorm, use_checkpoint = False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint
        if num_heads>10:
            drop_path = 1
        assert 0<=self.shift_size[0]<self.window_size[0], "shift size must be between 0 and window size"
        assert 0<=self.shift_size[1]<self.window_size[1], "shift size must be between 0 and window size"
        assert 0<=self.shift_size[2]<self.window_size[2], "shift size must be between 0 and window size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention3D(dim, window_size = self.window_size, num_heads = num_heads, qkv_bias = qkv_bias, 
                                      qk_scale = qk_scale, attn_drop = attn_drop, proj_drop = drop)
        self.drop_path = DropPath(drop_path) if drop_path>0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim*mlp_ratio)
        self.mlp  = MLP(in_features = dim, hidden_features = mlp_hidden_dim, act_layer = act_layer, drop = drop)
    def forward_path1(self,x,mask_matrix):
        B,D,H,W,C = x.shape
        window_size, shift_size = get_window_size((D,H,W), self.window_size, self.shift_size)

        x = self.norm1(x)

        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (window_size[0] - D% window_size[0]) % window_size[0]     
        pad_b = (window_size[1] - H% window_size[1]) % window_size[1]     
        pad_r = (window_size[2] - W% window_size[2]) % window_size[2]     
        x = F.pad(x,(0,0,pad_l,pad_r,pad_t,pad_b,pad_d0, pad_d1))
        _, Dp,Hp,Wp,_ = x.shape

        if any(i>0 for i in shift_size):
            shifted_x = torch.roll(x, shifts = (-shift_size[0], -shift_size[1], -shift_size[2]), dims = (1,2,3))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        x_windows = window_partition(shifted_x,window_size)
        attn_windows = self.attn(x_windows, mask = attn_mask)
        attn_windows = attn_windows.view(-1,*(window_size+(C,)))
        shifted_x = window_reverse(attn_windows,window_size, B, Dp,Hp,Wp)
        if any(i>0 for i in shift_size):
            x = torch.roll(shifted_x, shifts = (shift_size[0],shift_size[1], shift_size[2]), dims = (1,2,3))
        else:
            x = shifted_x
        
        if pad_d1 > 0 or pad_r>0 or pad_b > 0:
            x =x[:,:D,:H,:W,:].contiguous()
        return x
    def forward_path2(self,x):
        return self.drop_path(self.mlp(self.norm2(x)))
    
    def forward(self,x):
        _,d,_,w,h = x.shape
        x = x.permute(0,1,4,3,2).contiguous()
        a = get_window_size((d,h,w), self.window_size, self.shift_size)
        # print(a)
        # print(a.shape)
        window_size, shift_size = get_window_size((d,h,w), self.window_size, self.shift_size)
        mask_matrix = create_mask(d,h,w,window_size,shift_size, x.device, self.attn.qkv.weight.dtype)

        shortcut = x
        if self.use_checkpoint:
            x = checkpoint.checkpoint(self.forward_path1,x,mask_matrix)
        else:
            x = self.forward_path1(x,mask_matrix)
        x = shortcut + self.drop_path(x)
        if self.use_checkpoint:
            x = x + checkpoint.checkpoint(self.forward_path2,x)
        else:
            x = x+self.forward_path2(x)
        x = x.permute(0,1,4,3,2).contiguous()
        return x

class SwinTransformerBlock3D(nn.Module):
    def __init__(self, c1,num_frames = 5, num_layers = 2):
        super().__init__()
        num_heads = c1 //64
        window_size = 8
        self.num_frames = num_frames
        self.window_size = (num_frames,window_size,window_size)
        self.shift_size = (0,window_size//2,window_size//2)
        self.tr = nn.Sequential(*(SwinTransformerLayer3D(c1,num_heads,self.window_size,shift_size = (0,0,0) if i%2==0 else self.shift_size) for i in range(num_layers)))

    def reshape_frames(self,x, mode: int = 0):
        if mode == 0:
            b,c,h,w = x.shape
            b_new = b//self.num_frames
            # left = b%self.num_frames
            # x1 = x[-self.num_frames:,:,:,:]
            # x = x[:b_new*self.num_frames,:,:,:]
            x = x.reshape(b_new,self.num_frames,c,h,w)
            # x = torch.cat((x,x1))
        elif mode == 1:
            b,t,c,h,w = x.shape
            x = x.reshape(b*t,c,h,w)
        return x
    
    def save_attention_maps(self,ftmap,mode = 'before_attention'):
        b,t,c,h,w = ftmap.size()
        b0,t0 = 0,-1
        ftmap = ftmap[b0,t0].sum(dim = 0)
        ftmap /=c
        ftmap = ftmap.data.cpu().numpy()
        np.save(f'{mode}_{b0}_{t0}_feature_channel_{c}.npy',ftmap)
    
    def forward(self, x):
        x = self.reshape_frames(x,0)
        x = self.tr(x)
        x = self.reshape_frames(x,1)
        return x
