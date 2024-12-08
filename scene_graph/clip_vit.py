import torch
import torch.nn.functional as F
import math
from transformers import CLIPProcessor, CLIPVisionModel
from PIL import Image

def get_clip_vit_model(model_name, target_resolution=768):
    """
    Load CLIP ViT-B/32 and modify for target resolution
    """
    model = CLIPVisionModel.from_pretrained(model_name)

    # Original resolution and settings
    original_res = 224
    patch_size = 32
    
    # Calculate number of patches for original and target resolution
    orig_num_patches = (original_res // patch_size) ** 2
    new_num_patches = (target_resolution // patch_size) ** 2
    
    # Get original positional embeddings (excluding CLS token embedding)
    pos_emb = model.vision_model.embeddings.position_embedding.weight
    cls_emb = pos_emb[:1, :]
    pos_emb = pos_emb[1:, :]
    
    # Reshape to square grid
    side_len = int(math.sqrt(orig_num_patches))
    pos_emb = pos_emb.reshape(side_len, side_len, -1)
    pos_emb = pos_emb.unsqueeze(0)
    
    # Interpolate to new size
    new_side_len = int(math.sqrt(new_num_patches))
    pos_emb = pos_emb.permute(0, 3, 1, 2)
    pos_emb = F.interpolate(
        pos_emb, 
        size=(new_side_len, new_side_len), 
        mode='bicubic',
        align_corners=False
    )
    pos_emb = pos_emb.permute(0, 2, 3, 1).flatten(1, 2)
    
    # Recombine with CLS token embedding
    new_pos_emb = torch.cat([cls_emb.unsqueeze(0), pos_emb], dim=1)
    
    # Update model's positional embeddings
    model.vision_model.embeddings.position_embedding = torch.nn.Embedding.from_pretrained(new_pos_emb.squeeze(0))
    
    # Update position IDs to match new sequence length
    model.vision_model.embeddings.position_ids = torch.arange(new_num_patches + 1).unsqueeze(0)

    
    return model

def get_features(image_path, model, processor):
    """
    Get features in shape (batch_size, num_patches + 1, hidden_dim)
    """
    image = Image.open(image_path).convert('RGB')
    inputs = processor(images=image, return_tensors="pt").pixel_values

    print(inputs.shape)
    
    with torch.no_grad():
        outputs = model(inputs, output_hidden_states=True)
        # Get last hidden state: [batch_size, num_patches + 1, hidden_dim]
        features = outputs.last_hidden_state
        
    return features

# Example usage
if __name__ == "__main__":
    target_res = 768
    model, processor = get_clip_vit_model(target_resolution=target_res)
    
    # Process an image
    image_path = "results/output.jpg"
    features = get_features(image_path, model, processor)


    print(features.shape)
    
    # # Print shapes
    # num_patches = (target_res // 32) ** 2
    # print(f"Expected shape: [1, {num_patches + 1}, 768]")  # +1 for CLS token
    # print(f"Actual feature shape: {features.shape}")