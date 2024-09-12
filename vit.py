import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat, pack
from einops.layers.torch import Rearrange
from torch import einsum
import torch.nn.functional as F




# helper function
def exists(val):
	return val is not None

class LayerNorm(nn.Module):
	def __init__(self, dim):
		super().__init__()
		self.gamma = nn.Parameter(torch.ones(dim))
		# we don't want to update this
		self.register_buffer("beta", torch.zeros(dim))

	def forward(self, x):
		return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)




class SoftmaxAttention(nn.Module):
	def __init__(self, dim, num_heads=8, dim_head=64, dropout=0.0):
		super(SoftmaxAttention, self).__init__()

		self.dim = dim
		self.num_heads = num_heads
		self.dim_head = dim_head

		self.q = nn.Sequential(
			nn.Linear(dim, num_heads * dim_head, bias=False),
			nn.Dropout(dropout),
			Rearrange('b t (h d) -> b h t d', h=self.num_heads)
		)

		self.kv = nn.Sequential(
			nn.Linear(dim, 2 * num_heads * dim_head, bias=False),
			nn.Dropout(dropout),
			Rearrange('b t (kv h d) -> kv b h t d', d = self.dim_head, h = self.num_heads)
		)

		self.W_o = nn.Linear(num_heads * dim_head, dim)

		self.dropout = nn.Dropout(dropout)

		self.scale = dim_head ** -0.5

	def forward(self, x, context=None, causal_mask=None, context_mask=None):

		# prepare Q, K, V for attention

		q = self.q(x)

		if exists(context):
			k, v = self.kv(context)
		else:
			k, v = self.kv(x)
		
		# compute attention scores
		# Attention Scores = Q * K^T / sqrt(d_k)
		#  Q(b, h, t, d) * K(b, h, d, t) ->  (b, h, t, t)
		attn_scores = einsum('b h i d, b h d j -> b h i j', q * self.scale, k.transpose(-1, -2))
		
		# context mask used in Cross-Attention (encoder-decoder) and Self-Attention (encoder)
		if exists(context_mask):
			context_mask = rearrange(context_mask, 'b j -> b 1 1 j')
			attn_scores = attn_scores.masked_fill(~context_mask, -1e9)

		# causal mask used in Masked Multi-Head Attention (decoder)
		if exists(causal_mask):
			attn_scores = attn_scores.masked_fill(causal_mask, -1e9)
		attn_probs = torch.softmax(attn_scores, dim=-1)

		# Apply attention scores to V
		# (b, h, t, t) * V(b, h, t, d) -> (b, h, t, d)
		output = einsum('b h i j, b h j d -> b h i d', attn_probs, v)

		# combine heads
		output = rearrange(output, 'b h t d -> b t (h d)')
		output = self.W_o(output)
		output = self.dropout(output)
		return output

class Encoder(nn.Module):
	def __init__(self, dim, n_heads=8, d_head=64, depth=6, mlp_dim=3072, dropout=0.0):
		super().__init__()
	
		self.layers = nn.ModuleList([EncoderLayer(dim, n_heads, d_head, mlp_dim, dropout) for _ in range(depth)])
 
	def forward(self, x, context_mask=None):
		for layer in self.layers:
			x = layer(x, context_mask=context_mask)
		return x


class EncoderLayer(nn.Module):
	def __init__(self, dim, n_heads=8, d_head=64, mlp_dim=3072, dropout=0.0):
		super().__init__()

		self.self_attn = SoftmaxAttention(dim, n_heads, d_head, dropout)
		self.feed_forward = FeedForward(dim, mlp_dim)
		self.norm1 = LayerNorm(dim)
		self.norm2 = LayerNorm(dim)
		
	def forward(self, x, context_mask=None):
		x_norm = self.norm1(x)
		# self attention
		attn_out = self.self_attn(x=x_norm, context_mask=context_mask)

		# ADD & NORM
		x = attn_out + x
		x_norm = self.norm2(x)

		# feed forward
		fc_out = self.feed_forward(x_norm)

		# ADD
		x = fc_out + x
		return x



class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class ViT(nn.Module):
    def __init__(self, dim, image_size=256, patch_size = 16, n_heads = 12, d_head = 64, depth = 12, mlp_dim=3072, dropout=0.0):
        super(ViT, self).__init__()
        
        self.dim = dim
        self.patch_size = patch_size
        
        # number of features inside a patch
        self.patch_dim = patch_size * patch_size * 3
        
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size),
            LayerNorm(self.patch_dim),
            nn.Linear(self.patch_dim, dim),
            LayerNorm(dim)
        )

        num_patches = (image_size // patch_size) ** 2
        self.pos_enc =  nn.Parameter(torch.randn(1, num_patches, dim)) # 1 extra for class token

        self.encoder = Encoder(dim, n_heads, d_head, depth, mlp_dim, dropout)


    def forward(self, x):
        # (batch_size, channels, height, width) --> (batch_size, timesteps, features)
        x = self.to_patch_embedding(x)
		
        # add positional encoding
        x += self.pos_enc

        # transformer encoder
        x = self.encoder(x)

        return x

    



# model = ViT(1024, image_size=256, patch_size=32, depth=6, n_heads=16, mlp_dim=2048, dropout=0.0)
# img_batch = torch.randn(2, 3, 256, 256)
# out = model(img_batch)
# print(out.shape) # (b, num_classes)