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
from scene_graph.model import SceneGraphViT




if __name__ == "__main__":

    model = SceneGraphViT(
    dim=768,
    num_classes=100
    )
    

    # load the model
    ckpt = torch.load("outputs/scene-graph/checkpoints/scene-graph_run5.pt")

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
    valid_indices = (max_prob > 0.7) & (labels != 100)
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