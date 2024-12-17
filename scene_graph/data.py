import json
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils
from transformers import CLIPProcessor, CLIPVisionModel
from transformers import AutoImageProcessor, AutoModel
from types import SimpleNamespace

from torchvision.transforms import functional as TF
import torch.nn.functional as F

from .  import transforms as T

device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')

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



def y1y2x1x2_to_x1y1x2y2(y1y2x1x2):
	x1 = y1y2x1x2[2]
	y1 = y1y2x1x2[0]
	x2 = y1y2x1x2[3]
	y2 = y1y2x1x2[1]
	bbox = torch.tensor([x1, y1, x2, y2])
	return bbox

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def make_transform(img_size):

	normalize = T.Compose([
		T.ToTensor(),
		T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	])

	return T.Compose([
		T.RandomHorizontalFlip(),
		T.RandomResize([200, 300, 400, img_size], max_size=img_size),
		normalize,
		
	])



def make_image_list(dataset_path, type):
	imgs_list = []
	with open(os.path.join(dataset_path, 'json_dataset', f'annotations_{type}.json'), 'r') as f:
		annotations = json.load(f)
	sg_images = os.listdir(os.path.join(
		dataset_path, f'sg_{type}_images'))

	annotations_copy = annotations.copy()
	for ann in annotations.items():
		if(not annotations[ann[0]] or ann[0] not in sg_images):
			annotations_copy.pop(ann[0])

	for ann in annotations_copy.items():
		imgs_list.append(ann[0])
	return imgs_list


class VRDDataset(Dataset):
	"""VRD dataset."""

	def __init__(self, cfg, image_set):
		self.image_set = image_set

		self.dataset_path = cfg.dataset.params.root_path
		self.image_set = image_set
		self.image_size = cfg.dataset.preprocessing.resolution


		# self.transform = CLIPProcessor.from_pretrained(cfg.model.name)
		#   # Update processor's configuration to use new resolution
		# self.transform.feature_extractor.size = self.image_size
		# self.transform.feature_extractor.crop_size = self.image_size

		# read annotations file
		with open(os.path.join(self.dataset_path, 'json_dataset', f'annotations_{self.image_set}.json'), 'r') as f:
			self.annotations = json.load(f)
		with open(os.path.join(self.dataset_path, 'json_dataset', 'objects.json'), 'r') as f:
			self.all_objects = json.load(f)
		with open(os.path.join(self.dataset_path, 'json_dataset', 'predicates.json'), 'r') as f:
			self.predicates = json.load(f)

		self.root = os.path.join(
			self.dataset_path, f'sg_{self.image_set}_images')

		self.classes = self.all_objects.copy()
		self.preds = self.predicates.copy()
		# self.classes.insert(0, '__background__')
		self.preds.insert(0, 'unknown')

		self._class_to_ind = dict(zip(self.classes, range(len(self.classes))))
		self.ind_to_class = dict(zip(range(len(self.classes)), self.classes))
		self._preds_to_ind = dict(
			zip(self.preds, range(len(self.preds))))
		self.imgs_list = make_image_list(self.dataset_path, self.image_set)


		self.transform  = make_transform(self.image_size)

		# self.transform = transforms.Compose([
		# 	transforms.Resize((self.image_size)),
		# 	transforms.ToTensor(),
		# 	transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		# ])

		

		# self.processor = AutoImageProcessor.from_pretrained(cfg.model.name)

	def __len__(self):
		return len(self.imgs_list)

	def image_path_from_index(self, img_name):
		"""
		Construct an image path from the image's "index" identifier.
		"""
		image_path = os.path.join(self.dataset_path, f'sg_{self.image_set}_images',
								  img_name)
		assert os.path.exists(image_path), \
			'Path does not exist: {}'.format(image_path)
		return image_path

	def transform_boxes(self, boxes, original_size):
		"""
		Transform the bounding boxes to the scale of 224x224 image.
		"""

		orig_w, orig_h = original_size
		scale_w = self.image_size / orig_w
		scale_h = self.image_size  / orig_h
		transformed_boxes = []
	
		for box in boxes:
			x1, y1, x2, y2 = box
			x1 = int(x1 * scale_w)
			y1 = int(y1 * scale_h)
			x2 = int(x2 * scale_w)
			y2 = int(y2 * scale_h)
			transformed_boxes.append([x1, y1, x2, y2])
		return transformed_boxes

	def load_pascal_annotation(self, index, original_size):
		"""
		Load image and bounding boxes info from XML file in the PASCAL VOC
		format.
		"""
		sbj_boxes = []
		obj_boxes = []
		boxes = []
		labels = []
		preds = []
		# preds = []
		annotation = self.annotations[index]
		for spo in annotation:
			sbj_class = spo['subject']['category']
			sbj_y1y2x1x2 = spo['subject']['bbox']
			obj_class = spo['object']['category']
			obj_y1y2x1x2 = spo['object']['bbox']
			predicate = spo['predicate']

			# prepare bboxes for subject and object
			sbj_xyxy = y1y2x1x2_to_x1y1x2y2(sbj_y1y2x1x2)
			obj_xyxy = y1y2x1x2_to_x1y1x2y2(obj_y1y2x1x2)
			# transform the boxes
			# gt_sbj_bbox = self.transform_boxes([gt_sbj_bbox], original_size)[0]
			# gt_obj_bbox = self.transform_boxes([gt_obj_bbox], original_size)[0]
   
			# convert to cxcywh format
			# sbj_cxcywh = box_xyxy_to_cxcywh(torch.tensor(sbj_xyxy))
			# obj_cxcywh = box_xyxy_to_cxcywh(torch.tensor(obj_xyxy))
   
			# nomalize the boxes
			# w, h = original_size
			# sbj_cxcywh = sbj_cxcywh / torch.tensor([w, h, w, h], dtype=torch.float32)
			# obj_cxcywh = obj_cxcywh / torch.tensor([w, h, w, h], dtype=torch.float32)


			sbj_boxes.append(sbj_xyxy.tolist())
			obj_boxes.append(obj_xyxy.tolist())


			# prepare labels for subject and object
			# map to word
			sbj_class = self.all_objects[sbj_class]
			obj_class = self.all_objects[obj_class]
			predicate = self.predicates[predicate]
			# map to new index
			labels.append([self._class_to_ind[sbj_class],
						   self._class_to_ind[obj_class]])
			preds.append(self._preds_to_ind[predicate])

		return torch.tensor(sbj_boxes), torch.tensor(obj_boxes), labels, preds

	def __getitem__(self, index):
		img_name = self.imgs_list[index]
		img_path = self.image_path_from_index(img_name)
		img = Image.open(img_path)
		original_size = img.size
		sbj_boxes, obj_boxes , labels, preds = self.load_pascal_annotation(img_name, original_size)
		# img = self.transform(images=img, return_tensors="pt").pixel_values[0]

		target = {"sbj_boxes" : sbj_boxes, "obj_boxes": obj_boxes}
		transformed_img , target = self.transform(img, target)

		transformed_img_size = transformed_img.shape
		
		# add letterbox transform
		letterbox_padded_img = letterbox_transform(transformed_img, self.image_size)

		boxes = torch.stack((target["sbj_boxes"], target["obj_boxes"]), dim=1)
		
		# assert len(boxes) == len(
		# 	labels), "boxes and labels should be of equal length"

		return {'boxes': torch.tensor(boxes, dtype=torch.float32),
				'labels': torch.tensor(labels, dtype=torch.int64),
				'preds': torch.tensor(preds, dtype=torch.int64),
				'transformed_img_size' : transformed_img_size,
				'img': letterbox_padded_img
				}
	

def collater(data):
	# add letterbox transform
	imgs = [s['img'] for s in data]
	imgs = torch.stack(imgs, 0)
	annotations = [{"boxes": s['boxes'].to(device)} for s in data]
	for i, s in enumerate(data):
		annotations[i]['labels'] = s['labels'].to(device)
	for i, s in enumerate(data):
			annotations[i]['preds'] = s['preds'].to(device)
	for i, s in enumerate(data):
			annotations[i]['transformed_img_size'] = s['transformed_img_size']
	return imgs, annotations



def build_dataset(cfg):
	train_dataset = VRDDataset(cfg, "train")
	val_dataset = VRDDataset(cfg, "test")
	train_loader = DataLoader(train_dataset, batch_size=cfg.dataset.params.batch_size,
							  shuffle=cfg.dataset.params.shuffle, collate_fn=collater)
	val_loader = DataLoader(val_dataset, batch_size=cfg.dataset.params.batch_size,
							shuffle=False, collate_fn=collater)
	
	return train_loader, val_loader


if __name__ == '__main__':

	
	cfg = SimpleNamespace(
		dataset=SimpleNamespace(
			params=SimpleNamespace(
				root_path="/home/pranoy/Downloads/vrd",
				batch_size=2,
				shuffle=False,
			),
			preprocessing=SimpleNamespace(
				resolution=518
			)
		),
		model=SimpleNamespace(
			name="openai/clip-vit-base-patch32"
		)
	)


	dataset = VRDDataset(cfg, "train")
	dataloader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collater)
	for i, data in enumerate(dataloader):
		imgs, annotations = data
		# print(annotations)


		# visualize the image and the bounding boxes
		for j in range(len(imgs)):
			img = imgs[j].permute(1, 2, 0).cpu().numpy()
			img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
			img = np.clip(img, 0, 1)

			size = annotations[j]['transformed_img_size']

			plt.figure(figsize=(10, 10))
			plt.imshow(img)

			for boxes, labels in zip(annotations[j]['boxes'], annotations[j]['labels']):
				for box, label in zip(boxes, labels):
					c, h , w = size
					
					

					box = box_cxcywh_to_xyxy(box)
					box = box * torch.tensor([w, h, w, h], dtype=torch.float32)

					x1, y1, x2, y2 = box
     

    
					plt.text(x1, y1, dataset.ind_to_class[label.item()], fontsize=12, color='r')
					plt.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], color='r')
				
			
			plt.savefig(f'output_{i}_{j}.png')
	
	

	

		break
		# print(imgs)
		# print(annotations)
		# print(annotations['boxes'])
		# print(annotations['labels'])
		# print(annotations['preds'])
		# print(annotations['img'])
		# print(annotations['img'].shape)
		# print(annotations['boxes'].shape)
		# print(annotations['labels'].shape)
		# print(annotations['preds