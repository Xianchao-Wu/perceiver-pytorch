from math import pi, log
from functools import wraps

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cache_fn(f):
    cache = None
    @wraps(f)
    def cached_fn(*args, _cache = True, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache
    return cached_fn

def fourier_encode(x, max_freq, num_bands = 4, base = 2):
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x

    scales = torch.logspace(1., log(max_freq / 2) / log(base), num_bands, 
                            base = base, device = device, dtype = dtype)
    scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)]

    x = x * scales * pi
    x = torch.cat([x.sin(), x.cos()], dim=-1)
    x = torch.cat((x, orig_x), dim = -1)
    return x

# helper classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim = None):
        super().__init__()
        self.fn = fn # Attention's object
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context = normed_context)

        return self.fn(x, **kwargs)

class GEGLU(nn.Module):
    # https://arxiv.org/pdf/2002.05202.pdf
    # 有关于GEGLU函数的定义：GEGLU(x, W, V, b, c) = GELU(xW + b) ⊗ (xV + c)
    # GLU = gated linear units

    def forward(self, x):
        print('in geglu x={}'.format(x.shape)) # x=[1, 256, 4096]
        x, gates = x.chunk(2, dim = -1)
        # 按照最后一列切割成两个tensor
        print('in middle of geglu x={}, gates={}'.format(x.shape, gates.shape))
        # x=[1, 256, 2048], gates=[1, 256, 2048]

        out = x * F.gelu(gates)
        print('out geglu out={}'.format(out.shape))
        # out=[1, 256, 2048]
        return out

class FeedForward(nn.Module): # MLP，为何是dim -> dim*mult*2，而后居然是dim*mult -> dim???
    def __init__(self, dim, mult = 4, dropout = 0.): # dim=512, mult=4, dropout=0.0
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2), # 512, 512*4*2??? 
            GEGLU(), # 除非这里把最后一个维度的大小从512*4*2变换成了512*4->是的！
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim) # 512*4, 512
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, query_dim, context_dim = None, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim) 
        # 如果context_dim不为None，则返回context_dim；否则返回query_dim

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias = False) # 512, 64
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False) # 29, 64*2

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context = None, mask = None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        # qk^T/sqrt(dim_head) ? why dim_head?, should be inner_dim?

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            # from einops import rearrange, repeat
            # 强大的rearrange!

            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h = h)
            # 强大的repeat!

            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of (听不够，玩不够）
        attn = sim.softmax(dim = -1) # 执行softmax

        out = einsum('b i j, b j d -> b i d', attn, v) # 完成 softmax(qk^T/sqrt(d_q)) * v
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h) # 问题，有例子吗？为什么要有括号？
        return self.to_out(out) # 最后再增加一层linear layer

# main class

class Perceiver(nn.Module):
    def __init__(
        self,
        *,
        num_freq_bands, # 6
        depth, # 6
        max_freq, # 10.0
        freq_base = 2,
        input_channels = 3,
        input_axis = 2,
        num_latents = 512,
        cross_dim = 512,
        latent_dim = 512,
        cross_heads = 1,
        latent_heads = 8,
        cross_dim_head = 64,
        latent_dim_head = 64,
        num_classes = 1000,
        attn_dropout = 0.,
        ff_dropout = 0.,
        weight_tie_layers = False
    ):
        super().__init__()
        self.input_axis = input_axis # image, h and w, 2
        self.max_freq = max_freq # 10.0
        self.num_freq_bands = num_freq_bands # 6
        self.freq_base = freq_base # 2

        input_dim = input_axis * ((num_freq_bands * 2) + 1) + input_channels
        # 一上来就懵逼了，啥意思？ 2*(6*2+1)+3 = 29 TODO 干啥用的？
        # 回答：这里的，(num_freq_bands * 2 + 1 = 13代表的是，一个轴上的一个位置，使用一个13维度的向量表示；
        # input_axis * 13 = 26 代表的是两个轴，分别是13个维度的向量的话，得到的就是26维度
        # input_channels = 3，代表的是rgb这三个原始的channels
        # 故此，最初的input_dim就是，29了

        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim)) # (256=N, 512)
        # 输入的随机噪音，模拟的是Q

        get_cross_attn = lambda: PreNorm(latent_dim, Attention(latent_dim, input_dim,  # Attention(512, 29, 1, 64, 0.)
                                                               heads = cross_heads, 
                                                               dim_head = cross_dim_head, 
                                                               dropout = attn_dropout), 
                                         context_dim = input_dim)
        # 这块的代码类似于：
        # attention = Attention(latent_dim=512=Q的维度, 
        #                       input_dim=29=输入image的维度=3+26, 
        #                       heads=cross_heads=1, 
        #                       dim_head=64, 
        #                       dropout=0.)
        # prenorm = PreNorm(latent_dim=512, attention, context_dim=input_dim=29)
        # return prenorm

        # PreNorm (512, attention object, context_dim=29)
        # TODO lambda这里是啥用处？->相当于定义了一个无名函数，该函数首先定义一个Attention对象，其次定义个PreNorm对象

        get_cross_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim, dropout = ff_dropout)) 
        # FeedForward(512, dropout=0.)
        # PreNorm (512, FeedForward object)

        get_latent_attn = lambda: PreNorm(latent_dim, Attention(latent_dim, 
                                                                heads = latent_heads, 
                                                                dim_head = latent_dim_head, 
                                                                dropout = attn_dropout))
        get_latent_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim, dropout = ff_dropout))

        get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff = map(cache_fn, 
                                                                           (get_cross_attn, 
                                                                            get_cross_ff, 
                                                                            get_latent_attn, 
                                                                            get_latent_ff))

        self.layers = nn.ModuleList([])
        for i in range(depth):
            should_cache = i > 0 and weight_tie_layers
            cache_args = {'_cache': should_cache}

            self.layers.append(nn.ModuleList([
                get_cross_attn(**cache_args), # cross-attention (multi-head, alike softmax(QK^T/sqrt(d_q))V
                get_cross_ff(**cache_args), # cross-feed forward
                get_latent_attn(**cache_args), # self-attention (multi-head)
                get_latent_ff(**cache_args) # self-attention's feed forward
            ]))

        self.to_logits = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, num_classes)
        )
        print('done init perceiver.')

    def forward(self, data, mask = None):
        print('in forward of perceiver. data={}'.format(data.shape))
        b, *axis, _, device = *data.shape, data.device
        assert len(axis) == self.input_axis, 'input data must have the right number of axis'

        # calculate fourier encoded positions in the range of [-1, 1], for all axis
        # 位置编码：
        axis_pos = list(map(lambda size: torch.linspace(-1., 1., steps = size, device = device), axis)) # 哎呀，好nb啊！
        print('axis_pos={}'.format(len(axis_pos))) # 2， 图片的时候是2D
        for idx1, axis1 in enumerate(axis_pos):
            print('idx1={}, axis1.shape={}'.format(idx1, axis1.shape))
            # idx1=0, shape=torch.size([224])
            # idx2=0, shape=torch.size([224])

        pos = torch.stack(torch.meshgrid(*axis_pos), dim = -1)
        print('pos={}'.format(pos.shape)) # pos=torch.size([224, 224, 2])
        # 224*224个二维的点阵！
        
        # num_freq_bands=6 --> num_freq_bands*2 + 1 --> 13
        enc_pos = fourier_encode(pos, self.max_freq, self.num_freq_bands, base = self.freq_base)
        # 一个轴是13个值来表示一个position；这样的话，是2个轴(xy)所对应的26个值来表示平面上的一个position.
        # 而这样的2维度(xy)的点，有224*224个。
        print('enc_pos={}'.format(enc_pos.shape)) # [224, 224, 2, 13]
        
        enc_pos = rearrange(enc_pos, '... n d -> ... (n d)') # [224, 224, 26]
        print('enc_pos={}, after rearrange'.format(enc_pos.shape))
        enc_pos = repeat(enc_pos, '... -> b ...', b = b) # 不管...里面是啥玩意儿，我都要b(=batch)个重复的这样的...
        # 张量的dim=0新加一个batch_size维度
        print('enc_pos={}, after repeat'.format(enc_pos.shape)) # [1, 224, 224, 26]

        # concat to channels of data and flatten axis
        print('data={}, enc_pos={}'.format(data.shape, enc_pos.shape))
        # data=[1, 224, 224, 3]; enc_pos=[1, 224, 224, 26]
        data = torch.cat((data, enc_pos), dim = -1) # 把pixel encoding和position encoding串联起来
        print('before rearrange, data={}'.format(data.shape)) # [1, 224, 224, 29=3+26]
        data = rearrange(data, 'b ... d -> b (...) d') # TODO 干啥的？ -> 保留最初的维度和最后的维度不变，其他所有维度合并成一个维度
        # 这里的关键是 ... 到 (...)的部分
        print('after rearrange, data={}'.format(data.shape)) # [1, 224*224=50176, 29]
        
        x = repeat(self.latents, 'n d -> b n d', b = b) # 类似于从一个(n,d)扩展到b个(n,d)，其中b=batch_size
        # 按照batch.size对输入的原始的噪声进行duplicate
        print('init, x.shape={}'.format(x.shape))

        for idx, (cross_attn, cross_ff, latent_attn, latent_ff) in enumerate(self.layers):
            print('layer idx={}'.format(idx))
            x = cross_attn(x, context = data, mask = mask) + x # 每一层后边，都带有residual connection
            print('after cross attention, x.shape={}; input.context/data={}'.format(x.shape, data.shape))
            x = cross_ff(x) + x
            print('after cross feed-forward, x.shape={}'.format(x.shape))
            x = latent_attn(x) + x
            print('after latent attention, x.shape={}'.format(x.shape))
            x = latent_ff(x) + x
            print('after latent feed-forward, x.shape={}'.format(x.shape))

        '''
        init, x.shape=torch.Size([1, 256, 512])
        layer idx=0
        after cross attention, x.shape=torch.Size([1, 256, 512])
        after cross feed-forward, x.shape=torch.Size([1, 256, 512])
        after latent attention, x.shape=torch.Size([1, 256, 512])
        after latent feed-forward, x.shape=torch.Size([1, 256, 512])
        layer idx=1
        after cross attention, x.shape=torch.Size([1, 256, 512])
        after cross feed-forward, x.shape=torch.Size([1, 256, 512])
        after latent attention, x.shape=torch.Size([1, 256, 512])
        after latent feed-forward, x.shape=torch.Size([1, 256, 512])
        layer idx=2
        after cross attention, x.shape=torch.Size([1, 256, 512])
        after cross feed-forward, x.shape=torch.Size([1, 256, 512])
        after latent attention, x.shape=torch.Size([1, 256, 512])
        after latent feed-forward, x.shape=torch.Size([1, 256, 512])
        layer idx=3
        after cross attention, x.shape=torch.Size([1, 256, 512])
        after cross feed-forward, x.shape=torch.Size([1, 256, 512])
        after latent attention, x.shape=torch.Size([1, 256, 512])
        after latent feed-forward, x.shape=torch.Size([1, 256, 512])
        layer idx=4
        after cross attention, x.shape=torch.Size([1, 256, 512])
        after cross feed-forward, x.shape=torch.Size([1, 256, 512])
        after latent attention, x.shape=torch.Size([1, 256, 512])
        after latent feed-forward, x.shape=torch.Size([1, 256, 512])
        layer idx=5
        after cross attention, x.shape=torch.Size([1, 256, 512])
        after cross feed-forward, x.shape=torch.Size([1, 256, 512])
        after latent attention, x.shape=torch.Size([1, 256, 512])
        after latent feed-forward, x.shape=torch.Size([1, 256, 512])
        x.shape=torch.Size([1, 256, 512])
        '''

        print('x.shape={}'.format(x.shape)) # [1, 256, 512]
        x = x.mean(dim = -2)
        foutput = self.to_logits(x) 
        # layernorm -> linear (从latent_dim到num_classes) 还差一个softmax就可以转换成概率了
        return foutput
