import torch
import torch.nn as nn
from vit import ViT
from torch import einsum
from einops import rearrange, repeat, pack



class RelationshipAttention(nn.Module):
    def __init__(self, dim):
        super(RelationshipAttention, self).__init__()
        self.dim = dim
        # self.q = nn.Linear(dim, dim)
        # self.k = nn.Linear(dim, dim)

    def forward(self, q, k, top_k_instances=10, top_k_relationships=5):
        # q = self.q(q) query - subject
        # k = self.k(k) key - object

        scores = einsum('b i d, b d j -> b i j', q, k.transpose(-1, -2))
        scores = torch.softmax(scores, dim=-1)

        #  get diagonal
        diag = scores.diagonal(dim1=-2, dim2=-1)

        # relationship scores
        top_k_indices = torch.topk(diag, k=top_k_instances, dim=-1)[1]
        top_k_indices= torch.sort(top_k_indices, dim=-1, descending=False)[0]
        top_k_indices1 = top_k_indices.clone()
        scores = scores[torch.arange(scores.size(0)).unsqueeze(1), top_k_indices]
        top_k_indices = repeat(top_k_indices, 'b n ->  b r n', r=scores.shape[1])
        relationship_scores = scores.gather(-1, top_k_indices)

        # get top k relationships, subject-object indices
        top_k_rel_indices = torch.topk(relationship_scores, k=top_k_relationships, dim=-1)[1]
        split_shape = top_k_rel_indices.shape[-1] * top_k_rel_indices.shape[-2]
        top_k_rel_indices = relationship_scores.scatter(-1, top_k_rel_indices, -1)

        indices = torch.where(top_k_rel_indices == -1)
        indices = torch.stack(indices, dim=-1)
        indices = indices.split(split_shape, dim=-2)
        indices = torch.stack(indices)

        # map to original indices
        top_k_indices1 = repeat(top_k_indices1, 'b k -> b n k', n=split_shape)
        subject_object_indices = top_k_indices1.gather(-1, indices)

        # Replace the first index in subject_object_indices with batch ids
        batch_size = subject_object_indices.shape[0]
        batch_ids = torch.arange(batch_size).unsqueeze(-1).unsqueeze(-1).expand_as(subject_object_indices[:, :, :1])
       
        # subject and object indices
        subject_indices = torch.cat([batch_ids, subject_object_indices[:, :, 1:2]], dim=-1)
        object_indices = torch.cat([batch_ids, subject_object_indices[:, :, 2:3]], dim=-1)

        # get the subject and object embeddings
        subject_embeds = q[subject_indices[:, :, 0], subject_indices[:, :, 1]]
        object_embeds = q[object_indices[:, :, 0], object_indices[:, :, 1]]

        # add up the subject and object embeddings to get the relationship embeddings
        relationship_embeds = subject_embeds + object_embeds

        return subject_embeds, object_embeds, relationship_embeds

        

        
class SceneGraphViT(nn.Module):
    def __init__(self, 
        dim,
        image_size,
        patch_size,
        depth,
        n_heads,
        mlp_dim
        ):
        super(SceneGraphViT, self).__init__()
        self.vit = ViT(
            dim=dim,
            image_size=image_size,
            patch_size=patch_size,
            depth=depth,
            n_heads=n_heads,
            mlp_dim=mlp_dim
        )


        self.subject_head = nn.Linear(dim, dim)
        self.object_head = nn.Linear(dim, dim)

        self.relationship_attention = RelationshipAttention(dim)

    def forward(self, x):
        x = self.vit(x)
        subject_logits = self.subject_head(x)
        object_logits = self.object_head(x)

        # compute relationship attention
        subject_logits, object_logits, relationship_logits = self.relationship_attention(q=subject_logits, k=object_logits)

        return subject_logits, object_logits, relationship_logits



model = SceneGraphViT(
    dim=1024,
    image_size=256,
    patch_size=32,
    depth=12,
    n_heads=16,
    mlp_dim=2048
)

img_batch = torch.randn(4, 3, 256, 256)
subject_logits, object_logits, relationship_logits = model(img_batch)

print(subject_logits.shape, object_logits.shape, relationship_logits.shape)
