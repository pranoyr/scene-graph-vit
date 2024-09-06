import torch
import torch.nn as nn
from vit import ViT



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

    def forward(self, x):
        x = self.vit(x)
        return self.subject_head(x), self.object_head(x)



model = SceneGraphViT(
    dim=1024,
    image_size=256,
    patch_size=32,
    depth=12,
    n_heads=16,
    mlp_dim=2048
)

img_batch = torch.randn(2, 3, 256, 256)
subject_logits, object_logits = model(img_batch)
print(subject_logits.shape)
print(object_logits.shape)
