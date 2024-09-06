import torch
import torch.nn as nn



class SceneGraphViT(nn.Module):
    def __init__(self, num_classes):
        super(SceneGraphViT, self).__init__()
        self.vit = 