import os
import torch
import random
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm
from einops import rearrange
import logging
import torch.nn as nn
from transformers import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup
import torch



def get_scheduler(cfg, optimizer , **kwargs):
	warmup_steps = cfg.lr_scheduler.params.warmup_steps
	
	if cfg.lr_scheduler.name == "constant_with_warmup":
		scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)
	elif cfg.lr_scheduler.name == "cosine_with_warmup":
		scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=kwargs["decay_steps"])
	elif not cfg.lr_scheduler.name:
		scheduler = None
	else:
		raise ValueError(f"Unknown scheduler: {cfg.lr_scheduler.name}")

	return scheduler



def get_optimizer(cfg, params):
	lr = cfg.optimizer.params.learning_rate
	warmup_steps = cfg.lr_scheduler.params.warmup_steps
	beta1 = cfg.optimizer.params.beta1
	beta2 = cfg.optimizer.params.beta2
	decay_steps = cfg.lr_scheduler.params.decay_steps
	weight_decay = cfg.optimizer.params.weight_decay
	
	if cfg.optimizer.name == "adamw":
		optimizer = torch.optim.AdamW(params, lr=lr, betas=(beta1, beta2), weight_decay=weight_decay)
	elif cfg.optimizer.name == "adam":
		optimizer = torch.optim.Adam(params, lr=lr, betas=(beta1, beta2), weight_decay=weight_decay)
	else:
		raise ValueError(f"Unknown optimizer: {cfg.optimizer.name}")

	return optimizer



class BaseTrainer(object):
	def __init__(
		self, 
		cfg,
		model,
		dataloaders
		):

		self.cfg = cfg
		self.project_name = cfg.experiment.project_name
		self.exp_name = cfg.experiment.exp_name
		
		# init accelerator
		self.accelerator = Accelerator(
			mixed_precision=cfg.training.mixed_precision,
			gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
			log_with="wandb"
		)
		self.accelerator.init_trackers(
				project_name=cfg.experiment.project_name,
				init_kwargs={"wandb": {
				# "config" : cfg,
				"name" : self.exp_name}
		})

		# dataloaders
		self.train_dl , self.val_dl = dataloaders
		self.global_step = 0
		self.num_epoch = cfg.training.num_epochs
		self.gradient_accumulation_steps = cfg.training.gradient_accumulation_steps
		self.batch_size = cfg.dataset.params.batch_size
		self.max_grad_norm = cfg.training.max_grad_norm

		# logging details
		self.num_epoch = cfg.training.num_epochs
		self.save_every = cfg.experiment.save_every
		self.sample_every = cfg.experiment.sample_every
		self.log_every = cfg.experiment.log_every
		self.eval_every = cfg.experiment.eval_every

		# Training parameters
		self.decay_steps = cfg.lr_scheduler.params.decay_steps
		if not self.decay_steps:
			self.decay_steps = self.num_epoch * len(self.train_dl)
			
		# Checkpoint and generated images folder
		output_folder = f"outputs/{cfg.experiment.project_name}"
		self.checkpoint_folder = os.path.join(output_folder, 'checkpoints')
		os.makedirs(self.checkpoint_folder, exist_ok=True)
		
		self.image_saved_dir = os.path.join(output_folder, 'images')
		os.makedirs(self.image_saved_dir, exist_ok=True)


		logging.info(f"Train dataset size: {len(self.train_dl.dataset)}")
		logging.info(f"Val dataset size: {len(self.val_dl.dataset)}")

		# effective iteration considering gradient accumulation
		effective_batch_size = self.batch_size * self.gradient_accumulation_steps
		self.num_iters_per_epoch = math.ceil(len(self.train_dl.dataset) / effective_batch_size)
		self.total_iters = self.num_epoch * self.num_iters_per_epoch
		logging.info(f"Number of iterations per epoch: {self.num_iters_per_epoch}")
		logging.info(f"Total training iterations: {self.total_iters}")


	
	@property
	def device(self):
		return self.accelerator.device
	
	
	def train(self):
		raise NotImplementedError("Train method not implemented")
		

	def save_ckpt(self, rewrite=False):
		"""Save checkpoint"""

		filename = os.path.join(self.checkpoint_folder, f'{self.project_name}_{self.exp_name}_step_{self.global_step}.pt')
		if rewrite:
			filename = os.path.join(self.checkpoint_folder, f'{self.project_name}_{self.exp_name}.pt')
		
		checkpoint={
				'step': self.global_step,
				'state_dict': self.accelerator.unwrap_model(self.model).state_dict(),
				'optimizer': self.optim.state_dict(),
				'scheduler': self.scheduler.state_dict(),
				'config': self.cfg

			}

		self.accelerator.save(checkpoint, filename)
		logging.info("Saving checkpoint: %s ...", filename)
   
   
	def resume_from_checkpoint(self, checkpoint_path):
		"""Resume from checkpoint"""
		checkpoint = torch.load(checkpoint_path)
		self.global_step = checkpoint['step']
		self.model.load_state_dict(checkpoint['state_dict'])

		# resume optimizer and scheduler
		if 'optimizer' in checkpoint:
			self.optim.load_state_dict(checkpoint['optimizer'])
			logging.info("Optimizer loaded from checkpoint")
		if 'scheduler' in checkpoint:
			self.scheduler.load_state_dict(checkpoint['scheduler'])
			logging.info("Scheduler loaded from checkpoint")
		
		logging.info("Resume from checkpoint %s (global_step %d)", checkpoint_path, self.global_step)


	@torch.no_grad()
	def evaluate(self):
		raise NotImplementedError("Evaluate method not implemented")
	



class SceneGraphTrainer(BaseTrainer):
	def __init__(
		self, 
		cfg,
		model,
		dataloaders
		):
		super().__init__(cfg, model, dataloaders)

		# model, optimizer and scheduler
		self.model = model
		self.optim = get_optimizer(cfg, self.model.parameters()) # use different params for different layers if needed
		self.scheduler = get_scheduler(cfg, self.optim, decay_steps=self.decay_steps)

		# resume from checkpoint
		if cfg.experiment.resume_path_from_checkpoint:
			path = cfg.experiment.resume_path_from_checkpoint
			self.resume_from_checkpoint(path)


		(
			self.model,
			self.optim,
			self.scheduler,
			self.train_dl
	
		) = self.accelerator.prepare(
			self.model,
			self.optim,
			self.scheduler,
			self.train_dl
	 )
		

	def train(self):
		start_epoch=self.global_step//len(self.train_dl)
		self.model.train()
		for epoch in range(start_epoch, self.num_epoch):
			with tqdm(self.train_dl, dynamic_ncols=True, disable=not self.accelerator.is_main_process) as train_dl:
				for batch in train_dl:
					img , annots = batch
					img = img.to(self.device)

					with self.accelerator.accumulate(self.model):
						with self.accelerator.autocast():
							loss_dict = self.model(img, annots)
							weight_dict = self.model.criterion.weight_dict
							losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

							self.accelerator.backward(losses)
							if self.accelerator.sync_gradients and self.max_grad_norm:
								self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
							self.optim.step()
							if self.cfg.lr_scheduler.name:
								self.scheduler.step(self.global_step)
							self.optim.zero_grad()
							
							
						if not (self.global_step % self.save_every):
							self.save_ckpt(rewrite=True)
						
						# if not (self.global_step % self.eval_every):
						# 	self.model.eval()
						# 	outputs = torch.softmax(outputs, dim=1)
						# 	acc = (outputs.argmax(dim=1) == target).float().mean().item()
						# 	self.accelerator.log({"acc": acc}, step=self.global_step)
						# 	self.evaluate()
						# 	self.model.train()
		
						if not (self.global_step % self.gradient_accumulation_steps):
							lr = self.optim.param_groups[0]['lr']
							logs = {"total_loss": losses.item(), "lr": lr}
							for k, v in loss_dict.items():
								if k in weight_dict:
									logs[k] = v.item()
							self.accelerator.log(logs, step=self.global_step)
				
						self.global_step += 1
	  
					
		self.accelerator.end_training()        
		print("Train finished!")
  






