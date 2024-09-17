import torch
import torch.nn as nn
from vit import ViT
from torch import einsum
from einops import rearrange, repeat, pack
import torch.nn.functional as F
from transformers import CLIPTokenizer, CLIPTextModel

class TextEncoder:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.model = CLIPTextModel.from_pretrained(model_name)
        self.model.eval()  # Set the model to evaluation mode

    def encode(self, text):
        # Tokenize the input text
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        
        # Get the text features
        with torch.no_grad():
            outputs = self.model(**inputs)
            text_features = outputs.last_hidden_state.mean(dim=1)
        
        return text_features.squeeze()

    def encode_batch(self, texts):
        # Tokenize the batch of input texts
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        
        # Get the text features for the batch
        with torch.no_grad():
            outputs = self.model(**inputs)
            text_features = outputs.last_hidden_state.mean(dim=1)
        
        return text_features

# # Example usage
# if __name__ == "__main__":
#     encoder = TextEncoder()
    
#     # Single word encoding
#     word = "example"
#     vector = encoder.encode(word)
#     print(f"Vector for '{word}':")
#     print(vector)
#     print(f"Vector shape: {vector.shape}")
    
#     # Batch encoding
#     words = ["hello", "world", "python"]
#     vectors = encoder.encode_batch(words)
#     print(f"\nVectors for {words}:")
#     print(vectors)
#     print(f"Vectors shape: {vectors.shape}")



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

        # replace the first index with batch ids
        subject_object_indices = torch.cat((batch_ids, subject_object_indices[:, :, 1:]), dim=-1)

        # subject and object indices
        subject_indices = subject_object_indices[:, :, :-1]
        object_indices = torch.cat((batch_ids, subject_object_indices[:, :, -1:]), dim=-1)

        # get the subject and object embeddings
        subject_embeds = q[subject_indices[:, :, 0], subject_indices[:, :, 1]]
        object_embeds = q[object_indices[:, :, 0], object_indices[:, :, 1]]

        # add up the subject and object embeddings to get the relationship embeddings
        relationship_embeds = subject_embeds + object_embeds
        # layer norm
        relationship_embeds = F.layer_norm(relationship_embeds, normalized_shape=relationship_embeds.shape[-1:])

        return  subject_object_indices, relationship_embeds

        

        
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

        self.classifier = nn.Linear(dim, 100)
        self.bbox_mlp = nn.Sequential(
            nn.Linear(dim, 4),
            nn.ReLU()
        )

    def forward(self, x, annotations):
        x = self.vit(x)
        subject_logits = self.subject_head(x)
        object_logits = self.object_head(x)

        # compute relationship attention
        subject_object_indices, relationship_embeds = self.relationship_attention(q=subject_logits, k=object_logits)

        # object instances => subject == object
        object_indices = torch.where(subject_object_indices[:, :, 1] == subject_object_indices[:, :, 2])
        object_relationship_embeds = relationship_embeds[object_indices]

        bbox = self.bbox_mlp(object_relationship_embeds)
        cls_scores = self.classifier(object_relationship_embeds)


        print(f"bbox: {bbox.shape}")
        print(f"cls_scores: {cls_scores.shape}")

        return relationship_embeds



model = SceneGraphViT(
    dim=1024,
    image_size=256,
    patch_size=32,
    depth=12,
    n_heads=16,
    mlp_dim=2048
)

img_batch = torch.randn(4, 3, 256, 256)

gt_classes = [[1, 2, 3 ], [4, 5, 6]] # subject, object, relationship
gt_boxes = [[[1, 2, 3, 4],[5, 6, 7, 8]], [[13, 14, 15, 16], [21, 22, 23, 24]]] # subject, object boxes

annotations = {
    'gt_classes': gt_classes,
    'gt_boxes': gt_boxes
}

relationship_embeds = model(img_batch, annotations)

