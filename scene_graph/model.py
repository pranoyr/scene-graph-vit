import torch
import torch.nn as nn
from .vit import ViT
from torch import einsum
from einops import rearrange, repeat, pack
import torch.nn.functional as F
from transformers import CLIPTokenizer, CLIPTextModel
from .matcher import HungarianMatcher, SetCriterion
from transformers import AutoImageProcessor, Dinov2Model
from .clip_vit import get_clip_vit_model
import os
import logging

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

def exists(val):
    return val is not None

def parse_objects(annotations):
    targets =  []
    for i in range(len(annotations)):
        targets.append({
            'labels': annotations[i]['labels'].view(-1),
            'boxes': annotations[i]['boxes'].view(-1, 4)
        })

    return targets



class RelationshipAttention(nn.Module):
    def __init__(self, dim, top_k_instances=512, top_k_relationships=32):
        super(RelationshipAttention, self).__init__()
        self.dim = dim
        self.top_k_instances = top_k_instances
        self.top_k_relationships = top_k_relationships

        # self.q = nn.Linear(dim, dim)
        # self.k = nn.Linear(dim, dim)

    def forward(self, q, k):
        # q = self.q(q) query - subject
        # k = self.k(k) key - object

        device = q.device

        scores = einsum('b i d, b d j -> b i j', q, k.transpose(-1, -2))
        scores1 = scores.clone()
        scores = torch.softmax(scores, dim=-1)
    

        #  get diagonal
        diag = scores.diagonal(dim1=-2, dim2=-1)

        # relationship scores
        top_k_indices = torch.topk(diag, k=self.top_k_instances, dim=-1)[1]
        top_k_indices= torch.sort(top_k_indices, dim=-1, descending=False)[0]
        top_k_indices1 = top_k_indices.clone()
        scores = scores[torch.arange(scores.size(0)).unsqueeze(1), top_k_indices]
        top_k_indices = repeat(top_k_indices, 'b n ->  b r n', r=scores.shape[1])
        relationship_scores = scores.gather(-1, top_k_indices)

        max_int_value = 1e9
        relationship_scores.masked_fill_(torch.eye(relationship_scores.shape[1], relationship_scores.shape[2], dtype=bool, device=device).unsqueeze(0).expand(relationship_scores.shape[0], -1, -1), max_int_value)


        # get top k relationships, subject-object indices
        top_k_rel_indices = torch.topk(relationship_scores, k=self.top_k_relationships, dim=-1)[1]
        # add all diag indices

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
        batch_ids = torch.arange(batch_size).unsqueeze(-1).unsqueeze(-1).expand_as(subject_object_indices[:, :, :1]).to(device)

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

        return scores1,  subject_object_indices, relationship_embeds

        

class MLP(nn.Module):
    def __init__(self, dim):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.LayerNorm(dim)
        )

    def forward(self, x):
        return self.net(x)
    
# 2 Layer MLP
class MLP2(nn.Module):
    def __init__(self, dim):
        super(MLP2, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)
        )

    def forward(self, x):
        return self.net(x)
    

class VitBackbone(nn.Module):
    def __init__(self, cfg
        ):
        super(VitBackbone, self).__init__()

        self.cfg = cfg
        
        if cfg.model.name == "vit":
            self.model = ViT(
                dim=cfg.model.dim,
                image_size=cfg.dataset.preprocessing.resolution,
                patch_size=cfg.model.patch_size,
                depth=cfg.model.depth,
                n_heads=cfg.model.n_heads,
                mlp_dim=cfg.model.mlp_dim
            )

        model_name_prefix = cfg.model.name.split(os.sep)[0]
        if model_name_prefix == "clip":
            self.model = get_clip_vit_model(cfg.model.name, target_resolution=cfg.dataset.preprocessing.resolution)
        elif model_name_prefix == "facebook":
            self.model = Dinov2Model.from_pretrained(cfg.model.name)

        if cfg.model.freeze: 
            for param in self.model.parameters():
                    param.requires_grad = False
            logging.info("Model is frozen")

    def forward(self, x):
        if self.cfg.model.name == "vit":
                x = self.model(x)
        else:
                x = self.model(x, output_hidden_states=True)
                # Get last hidden state: [batch_size, num_patches + 1, hidden_dim]
                x = x.last_hidden_state
                x = x[:, 1:, :]
        return x
            
        
        
class SceneGraphViT(nn.Module):
    def __init__(self, 
        cfg,
        ):
        super(SceneGraphViT, self).__init__()

        dim = cfg.model.dim
        num_classes = cfg.model.num_classes
      
        self.vit = VitBackbone(cfg)
        
        self.subject_head = MLP(dim)
        self.object_head = MLP(dim)

        self.mlp = MLP2(dim)

        self.relationship_attention = RelationshipAttention(dim, top_k_instances=cfg.model.top_k_instances, top_k_relationships=cfg.model.top_k_relationships)

        self.matcher = HungarianMatcher(cfg.loss_weight.ce, cfg.loss_weight.bbox, cfg.loss_weight.giou)


        weight_dict = {'loss_ce': cfg.loss_weight.ce, 'loss_bbox': cfg.loss_weight.bbox}
        weight_dict['loss_giou'] = cfg.loss_weight.giou

        weight_dict['loss_scores'] = cfg.loss_weight.scores

        losses = ['labels', 'boxes', 'cardinality']
        self.criterion = SetCriterion(num_classes, matcher=self.matcher, weight_dict=weight_dict,
                             eos_coef=0.1, losses=losses)

        self.classifier = nn.Linear(dim, num_classes + 1)
        self.bbox_mlp = nn.Sequential(
            nn.Linear(dim, 4),
            nn.ReLU()
        )

    def forward(self, x, annotations=None):

        b = len(x)
        
        x = self.vit(x)
    
        subject_logits = self.subject_head(x)
        subject_logits = x + subject_logits

        object_logits = self.object_head(x)
        object_logits = x + object_logits

        # compute relationship attention ,  relationship_embeds - Rij => (b, number of relationships, dim)
        scores, subject_object_indices, relationship_embeds = self.relationship_attention(q=subject_logits, k=object_logits)

        # 2 layer MLP
        relationship_embeds = self.mlp(relationship_embeds)

        # object instances => subject == object
        object_indices = torch.where(subject_object_indices[:, :, 1] == subject_object_indices[:, :, 2])


        object_relationship_embeds = relationship_embeds[object_indices]
        object_relationship_embeds = rearrange(object_relationship_embeds, '(b n) d -> b n d', b=b)

        
        bbox = self.bbox_mlp(object_relationship_embeds)
        logits = self.classifier(object_relationship_embeds)

        logits_softmax = torch.softmax(logits, dim=-1)
        preds_topk = torch.topk(logits_softmax, k=1, dim=-1)[0]

        if not exists(annotations):
            return logits_softmax, bbox

        outputs = {}
        outputs['pred_logits'] = logits
        outputs['pred_boxes'] = bbox

        targets = parse_objects(annotations)

        # loss function
        matched_indices , loss = self.criterion(outputs, targets)
        
        # loss function for relationship scores (self-supervised with class-logits)
        num_objects = outputs['pred_logits'].shape[1]
        indx = []
        s = 0
        for i, j in matched_indices:
            for idx in i:
                indx.append((object_indices[0][idx + s].item(), object_indices[1][idx + s].item()))
            s += num_objects
        indx = torch.tensor(indx)

        matched_subject_object_indices = subject_object_indices[indx[:, 0], indx[:, 1]]

        targets = torch.zeros_like(scores)
        
        #TODO  set the matched indices to prob
        # targets[matched_subject_object_indices[:, 0], matched_subject_object_indices[:, 1], matched_subject_object_indices[:, 2]] = 1
        for i, (batch_idx, row_idx, col_idx) in enumerate(matched_subject_object_indices):
            # Get batch number and position within that batch
            b = batch_idx.item()
            # Find position within the batch by counting previous occurrences of this batch number
            pos = (matched_subject_object_indices[:i+1, 0] == b).sum() - 1
            # Assign the corresponding value from preds_topk
            targets[b, row_idx, col_idx] = preds_topk[b][pos].item()


        loss_scores = torch.nn.functional.binary_cross_entropy_with_logits(scores, targets)
        loss.update({'loss_scores': loss_scores})
        return loss





if __name__ == "__main__":

    # model = SceneGraphViT(
    # dim=1024,
    # image_size=256,
    # patch_size=32,
    # depth=12,
    # n_heads=16,
    # mlp_dim=2048
    # )



    img_batch = torch.randn(2, 3, 256, 256)
    annotations = [{'boxes': torch.tensor([[[349.,  16., 436., 208.],
            [139., 138., 756., 637.]],

            [[349.,  16., 436., 208.],
            [368.,  14., 423.,  37.]],

            [[349.,  16., 436., 208.],  # sampel 1
            [139., 138., 756., 637.]],

            [[ 74., 301., 152., 537.],
            [ 79., 332., 152., 424.]],

            [[ 74., 301., 152., 537.],
            [139., 138., 756., 637.]]]), 
            'labels': torch.tensor([[ 0, 68],
            [ 0, 13],
            [ 0, 68],
            [ 0,  6],
            [ 0, 68]]), 'preds': torch.tensor([36,  2,  1,  3,  4])}, 
            
            
            {'boxes': torch.tensor([[[1.0000e+00, 2.0000e+00, 1.0150e+03, 1.5200e+02],
            [6.0900e+02, 1.3100e+02, 7.5900e+02, 3.8400e+02]],

            [[6.0900e+02, 1.3100e+02, 7.5900e+02, 3.8400e+02],
            [6.1300e+02, 2.0100e+02, 6.3800e+02, 2.2000e+02]],

            [[6.1300e+02, 2.0100e+02, 6.3800e+02, 2.2000e+02],          # sample 2
            [6.1300e+02, 1.9800e+02, 6.5100e+02, 2.2700e+02]],

            [[7.3300e+02, 3.6700e+02, 7.5900e+02, 4.0000e+02],
            [6.0900e+02, 1.3100e+02, 7.5900e+02, 3.8400e+02]]]), 
            'labels': torch.tensor([[14,  0],
            [ 0, 85],
            [85, 57],
            [33,  0]]), 'preds': torch.tensor([11, 28, 23,  1])}]

    loss = model(img_batch, annotations)
    print(loss)

