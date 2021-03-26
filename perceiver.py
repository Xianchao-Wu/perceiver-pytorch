import torch
from perceiver_pytorch import Perceiver

model = Perceiver(
    input_channels = 3,          # number of channels for each token of the input
    input_axis = 2,              # number of axis for input data (2 for images, 3 for video)
    num_freq_bands = 6,          # number of freq bands, with original value (2 * K + 1) - 傅里叶位置编码用
    max_freq = 10.,              # maximum frequency, hyperparameter depending on how fine the data is
    depth = 6,                   # depth of net
    num_latents = 256,           # number of latents, or induced set points, or centroids. 
                                 #     different papers giving it different names
    cross_dim = 512,             # cross attention dimension
    latent_dim = 512,            # latent dimension
    cross_heads = 1,             # number of heads for cross attention. paper said 1
    latent_heads = 8,            # number of heads for latent self attention, 8
    cross_dim_head = 64,
    latent_dim_head = 64,
    num_classes = 1000,          # output number of classes
    attn_dropout = 0.,
    ff_dropout = 0.,
    weight_tie_layers = False    # whether to weight tie layers (optional, as indicated in the diagram)
)

# model's parameters and their names?

for name, param in model.named_parameters():
	print(name,param.requires_grad, param.shape)
	#param.requires_grad=False

# batch_size=1, height=224, width=224 and channels=3(RGB)
img = torch.randn(1, 224, 224, 3) # 1 imagenet image, pixelized

imgout = (model(img)) # (1, 1000)
print('input.shape={}, output.shape={}'.format(img.shape, imgout.shape))
