import torch
import torch.nn as nn
from scene_graph.vit import ViT
from torch import einsum
from einops import rearrange, repeat, pack
import torch.nn.functional as F
from transformers import CLIPTokenizer, CLIPTextModel
import os
from omegaconf import DictConfig, ListConfig, OmegaConf
from scene_graph.matcher import HungarianMatcher, SetCriterion
from transformers import AutoImageProcessor, Dinov2Model
from PIL import Image
import json
from torchvision import transforms
import cv2
import numpy as np
import sys
from scene_graph.model import SceneGraphViT
from types import SimpleNamespace
import matplotlib.pyplot as plt
import random
import scene_graph.transforms as T

def letterbox_transform(image, target_size, color=(114, 2, 114)):
    """
    Resizes and pads an image to maintain aspect ratio using letterbox padding,
    without upscaling if the original image is smaller than the target size,
    and places the image at the top-left of the target canvas.
    """
    target_size = (target_size, target_size) if isinstance(target_size, int) else target_size
    orig_h, orig_w = image.shape[1], image.shape[2]
    target_h, target_w = target_size

    # Compute scale factor, but don't upscale
    scale = min(target_w / orig_w, target_h / orig_h, 1.0)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)

    # Resize only if needed
    if (new_w != orig_w) or (new_h != orig_h):
        resized_image = TF.resize(image, (new_h, new_w))
    else:
        resized_image = image

    # Calculate padding for top-left placement
    pad_w = target_w - new_w
    pad_h = target_h - new_h

    pad_left = 0
    pad_top = 0
    pad_right = pad_w
    pad_bottom = pad_h

    # Apply padding in left, right, top, bottom order
    padded_image = F.pad(resized_image, (pad_left, pad_right, pad_top, pad_bottom),
                         mode='constant', value=color[0])

    return padded_image




def make_transform():
    
	normalize = T.Compose([
		T.ToTensor(),
		T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	])

	return T.Compose([
		# T.RandomHorizontalFlip(),
		T.RandomResize([518], max_size=518),
		normalize,
		
	])



# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]


# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


def plot_results(pil_img, prob, boxes):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), COLORS * 100):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        cl = p.argmax()
        text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    # save the image    
    plt.savefig("results/output.jpg")



def get_config():
	"""	Creates a config object from the yaml file and the cli arguments
	"""
	cli_conf = OmegaConf.from_cli()

	yaml_conf = OmegaConf.load(cli_conf.config)
	conf = OmegaConf.merge(yaml_conf, cli_conf)
	return conf




if __name__ == "__main__":

    

    cfg = get_config()
    model = SceneGraphViT(cfg)


    filename = os.path.join("outputs/scene-graph/checkpoints", f'{cfg.experiment.project_name}_{cfg.experiment.exp_name}.pt')

    print(f"Loading model from {filename}")
    

    # load the model
    ckpt = torch.load(filename)

    model.load_state_dict(ckpt['state_dict'])

    model.eval()

    dataset_path= cfg.dataset.params.root_path


    with open(os.path.join(dataset_path, 'json_dataset', 'objects.json'), 'r') as f:
        CLASSES = json.load(f)

    # transform = transforms.Compose([
	# 		transforms.Resize((cfg.dataset.preprocessing.resolution, cfg.dataset.preprocessing.resolution)),
	# 		transforms.ToTensor(),
	# 		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	# 	])
 
    transform = make_transform()


    image_dir_path = cfg.dataset.params.root_path + "/sg_train_images"
    # image_dir_path = "data"
    # Get a list of all image files in the directory
    image_files = [f for f in os.listdir(image_dir_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files:
        print("No image files found in the directory.")
        sys.exit(1)

    # Randomly select an image file
    image_path = os.path.join(image_dir_path, random.choice(image_files))
    print(f"Selected image: {image_path}")



    img_org = Image.open(image_path)
    transformed_img, _ = transform(img_org, target={}) 
    
    print("Transformed image shape:", transformed_img.shape)

    # add letterbox
    lettertbox_img = letterbox_transform(transformed_img, 518).unsqueeze(0)
    
    cv2.imwrite("results/letterbox.jpg", lettertbox_img[0].permute(1, 2, 0).numpy() * 255)
    
    print("Letterbox image shape:", lettertbox_img.shape)
    
 
    softmax_logits, bboxes = model(lettertbox_img)


    probas = softmax_logits[0, :, :-1]
    keep = probas.max(-1).values > 0.7
    
    filtered_bbox = rescale_bboxes(bboxes[0, keep], img_org.size)
    filtered_labels = probas[keep]
    
    
    plot_results(img_org, filtered_labels, filtered_bbox)