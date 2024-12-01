import torch
import torch.nn as nn
from scene_graph.vit import ViT
from torch import einsum
from einops import rearrange, repeat, pack
import torch.nn.functional as F
from transformers import CLIPTokenizer, CLIPTextModel
import os
from scene_graph.matcher import HungarianMatcher, SetCriterion
from transformers import AutoImageProcessor, Dinov2Model
from PIL import Image
import json
from torchvision import transforms
import cv2
import numpy as np
import sys

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



def parse_objects(annotations):
    targets =  []
    for i in range(len(annotations)):
        targets.append({
            'labels': annotations[i]['labels'].view(-1),
            'boxes': annotations[i]['boxes'].view(-1, 4)
        })

    return targets



class RelationshipAttention(nn.Module):
    def __init__(self, dim):
        super(RelationshipAttention, self).__init__()
        self.dim = dim
        # self.q = nn.Linear(dim, dim)
        # self.k = nn.Linear(dim, dim)

    def forward(self, q, k, top_k_instances=100, top_k_relationships=5):
        # q = self.q(q) query - subject
        # k = self.k(k) key - object

        device = q.device

        scores = einsum('b i d, b d j -> b i j', q, k.transpose(-1, -2))
        scores = torch.softmax(scores, dim=-1)
        scores1 = scores.clone()


        #  get diagonal
        diag = scores.diagonal(dim1=-2, dim2=-1)

        # relationship scores
        top_k_indices = torch.topk(diag, k=top_k_instances, dim=-1)[1]
        top_k_indices= torch.sort(top_k_indices, dim=-1, descending=False)[0]
        # print(top_k_indices)
        top_k_indices1 = top_k_indices.clone()
        scores = scores[torch.arange(scores.size(0)).unsqueeze(1), top_k_indices]
        top_k_indices = repeat(top_k_indices, 'b n ->  b r n', r=scores.shape[1])
        relationship_scores = scores.gather(-1, top_k_indices)
        # print(relationship_scores.shape)

        max_int_value = 1e9
        relationship_scores.masked_fill_(torch.eye(relationship_scores.shape[1], relationship_scores.shape[2], dtype=bool, device=device).unsqueeze(0).expand(relationship_scores.shape[0], -1, -1), max_int_value)


        # get top k relationships, subject-object indices
        top_k_rel_indices = torch.topk(relationship_scores, k=top_k_relationships, dim=-1)[1]
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

        return  scores1, subject_object_indices, relationship_embeds

        

        
class SceneGraphViT(nn.Module):
    def __init__(self, 
        dim=1024,
        image_size=256,
        patch_size=32,
        depth=12,
        n_heads=16,
        mlp_dim=2048,
        num_classes=100
        ):
        super(SceneGraphViT, self).__init__()

        # self.vit = ViT(
        #     dim=dim,
        #     image_size=image_size,
        #     patch_size=patch_size,
        #     depth=depth,
        #     n_heads=n_heads,
        #     mlp_dim=mlp_dim
        # )

 
        self.vit = Dinov2Model.from_pretrained("facebook/dinov2-base")
        # freeze the model
        for param in self.vit.parameters():
            param.requires_grad = False

        self.subject_head = nn.Linear(dim, dim)
        self.object_head = nn.Linear(dim, dim)

        self.relationship_attention = RelationshipAttention(dim)

        self.matcher = HungarianMatcher()


        weight_dict = {'loss_ce': 1, 'loss_bbox': 5}
        weight_dict['loss_giou'] = 2

        losses = ['labels', 'boxes', 'cardinality']
        self.criterion = SetCriterion(num_classes, matcher=self.matcher, weight_dict=weight_dict,
                             eos_coef=0.1, losses=losses)

        self.classifier = nn.Linear(dim, num_classes + 1)
        self.bbox_mlp = nn.Sequential(
            nn.Linear(dim, 4),
            nn.ReLU()
        )

    def forward(self, x):

        b = len(x)

        x = self.vit(x)
        x = x.last_hidden_state
        
        subject_logits = self.subject_head(x)
        object_logits = self.object_head(x)

        # compute relationship attention ,  relationship_embeds - Rij => (b, number of relationships, dim)
        scores, subject_object_indices, relationship_embeds = self.relationship_attention(q=subject_logits, k=object_logits)

        # object instances => subject == object
        object_indices = torch.where(subject_object_indices[:, :, 1] == subject_object_indices[:, :, 2])


        object_relationship_embeds = relationship_embeds[object_indices]
        object_relationship_embeds = rearrange(object_relationship_embeds, '(b n) d -> b n d', b=b)


        bbox = self.bbox_mlp(object_relationship_embeds)
        logits = self.classifier(object_relationship_embeds)

        return logits, bbox




if __name__ == "__main__":

    model = SceneGraphViT(
    dim=768,
    image_size=256,
    patch_size=32,
    depth=12,
    n_heads=16,
    mlp_dim=2048,
    num_classes=100
    )
    

    # load the model
    ckpt = torch.load("outputs/scene-graph/checkpoints/scene-graph_run3.pt")

    model.load_state_dict(ckpt['state_dict'])

    model.eval()

    dataset_path= "/home/pranoy/Downloads/vrd"


    with open(os.path.join(dataset_path, 'json_dataset', 'objects.json'), 'r') as f:
        all_objects = json.load(f)

    print(all_objects)


    # Create a dictionary to map integer labels to object names
    int_to_object = {i: obj for i, obj in enumerate(all_objects)}


    transform = transforms.Compose([
			transforms.Resize((256, 256)),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		])


    if len(sys.argv) != 2:
        print("Usage: python inference.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    img = Image.open(image_path)
    img_draw = np.array(img)
    img_draw = cv2.resize(img_draw, (256, 256))
    img = transform(img).unsqueeze(0)
    logits, bbox = model(img)


    print(logits)
    print(bbox)

    # print(logits.shape)
    # Get the most probable boxes
    prob = F.softmax(logits, dim=-1)
    max_prob, labels = torch.max(prob, dim=-1)




    # print(logits.shape, bbox.shape)
    # Filter out the labels with probability less than 0.5 and ignore the label 100
    valid_indices = (max_prob > 0.75) & (labels != 100)
    filtered_labels = labels[valid_indices]
    filtered_bbox = bbox[valid_indices]

    print("Filtered Labels:", filtered_labels)
    print("Filtered Bounding Boxes:", filtered_bbox)
    predicted_labels = [int_to_object[label.item()] for label in filtered_labels]
    print("Predicted Labels:", predicted_labels)

    # Convert the image tensor to a numpy array and transpose it to HWC format
    # Draw the bounding boxes on the image
    for box, label in zip(filtered_bbox, predicted_labels):
        x1, y1, x2, y2 = box.int().tolist()
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_draw, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Save or display the image
    cv2.imwrite("results/output.jpg", img_draw)
    # cv2.imshow("Annotated Image", img_np)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()