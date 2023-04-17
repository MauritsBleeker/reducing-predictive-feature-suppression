import torch
import torch.nn as nn
from models.image_encoder import ImageEncoder
from models.caption_encoder import CaptionEncoder
from models.target_decoder import TargetDecoder
from models.input_decoder import InputDecoder
from models.TERN.model import ImageCaptionEncoder as TERN
from models.VSRN.image_encoder import ImageEncoder as VSRNImageEncoder
import logging


class ImageCaptionEncoder(nn.Module):

	def __init__(self, config, vocab, init_weights=True):
		"""

		:param config: config class
		:param vocab: vocab file (dictionary)
		"""
		super(ImageCaptionEncoder, self).__init__()

		self.config = config

		self.logging_gradients = []

		if 'vsrn' in self.config.model and self.config.model.vsrn.use_model:
			self.image_encoder = VSRNImageEncoder(config=self.config)
			self.caption_encoder = CaptionEncoder(word2idx=vocab.word2idx, config=self.config, init_weights=init_weights)
			self.logging_gradients = [self.caption_encoder, self.image_encoder]

		elif 'TERN' in self.config.model and self.config.model.tern.use_model:
			self.tern = TERN(config=self.config)
			self.logging_gradients = [self.tern]

		else:
			self.image_encoder = ImageEncoder(config=self.config, init_weights=init_weights)
			self.caption_encoder = CaptionEncoder(word2idx=vocab.word2idx, config=self.config, init_weights=init_weights)
			self.logging_gradients = [self.caption_encoder, self.image_encoder]

		if self.config.model.target_decoder.decode_target:

			if self.config.model.target_decoder.input_decoding:
				self.input_decoder = InputDecoder(
					output_size=len(vocab.word2idx),
					config=self.config
				)
			else:
				self.target_decoder = TargetDecoder(
					in_features=self.config.model.embed_dim,
					hidden_features=self.config.model.target_decoder.hidden_features,
					reconstruction_dim=self.config.model.target_decoder.reconstruction_dim
				)

		self.iteration = 0

		self.img_encoder_device = self.cap_encoder_device = 'cpu'

		self.to_devide()

	def forward(self, images, captions, cap_lengths, boxes=None):
		"""
		Forward pass

		:param images: input images
		:param captions: input captions
		:param cap_lengths: length of the captions
		:return: latent of the images, captions and reconstructions
		"""
		self.iteration += 1

		if 'TERN' in self.config.model and self.config.model.tern.use_model:
			z_images, z_captions = self.tern(images, captions, cap_lengths, boxes)
		else:
			z_images = self.image_encoder(images)
			z_captions = self.caption_encoder(captions, cap_lengths, device=self.cap_encoder_device)

		reconstructions = None

		if self.config.model.target_decoder.decode_target:
			if self.config.model.target_decoder.input_decoding:
				reconstructions = self.input_decoder(z_captions=z_captions, targets=captions, lengths=cap_lengths)
			else:
				reconstructions = self.target_decoder(z_captions)

		return z_images, z_captions, reconstructions

	def finetune(self, image_encoder=True, caption_encoder=True):
		"""
		Start fine tuning encoders, if applicable
		:param image_encoder:
		:param caption_encoder:
		:return:
		"""

		logging.info("Start fine tuning encoders")

		if caption_encoder:
			self.caption_encoder.finetune()
		if image_encoder:
			self.image_encoder.finetune()

	def load_checkpoint(self, checkpoint_file, strict=False):
		"""
		Load model checkpoint
		:param checkpoint_file: PyTorch checkpoint file
		:param strict: use strict param loading
		:return: None
		"""

		self.load_state_dict(checkpoint_file['model'], strict=strict)

	def to_devide(self):
		"""
		To GPU, if available
		:return:
		"""
		if torch.cuda.is_available():
			# does not work for TERN
			if torch.cuda.device_count() == 2:
				logging.info("Using two GPUs")

				self.img_encoder_device = 'cuda:0'
				self.cap_encoder_device = 'cuda:1'

				self.caption_encoder.to(self.cap_encoder_device)
				self.image_encoder.to(self.img_encoder_device)

				if self.config.model.target_decoder.decode_target:
					if self.config.model.target_decoder.input_decoding:
						self.input_decoder.to(self.cap_encoder_device)
					else:
						self.target_decoder.to(self.cap_encoder_device)

			else:
				logging.info("Using one GPU")
				self.to('cuda')

				self.img_encoder_device = self.cap_encoder_device = 'cuda'

		else:
			logging.info("Using CPU")
			self.to('cpu')
