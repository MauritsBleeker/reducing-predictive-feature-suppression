"""
Reference code:
"""
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.init
import logging

from models.TERN.caption_encoder import EncoderTextBERT
from models.TERN.image_encoder import ImageEncoder
from utils.tern import l2norm, Aggregator


class JointTextImageTransformerEncoder(nn.Module):
	"""
	This is a bert caption encoder - transformer image encoder (using bottomup features).
	If process the encoder outputs through a transformer, like VilBERT and outputs two different graph embeddings
	"""

	def __init__(self, config):

		super().__init__()

		self.config = config
		self.txt_enc = EncoderTextBERT(config, post_transformer_layers=self.config.model.tern.text_model.layers)

		visual_feat_dim = self.config.model.tern.image_model.feat_dim
		caption_feat_dim = self.config.model.tern.text_model.word_dim
		dropout = self.config.model.tern.model.dropout
		layers = self.config.model.tern.model.layers
		embed_size = self.config.model.tern.model.embed_size

		self.img_enc = ImageEncoder(
			self.config.model.tern.image_model.transformer_layers,
			self.config.model.tern.image_model.feat_dim,
			self.config.model.tern.model.embed_size,
			n_head=4,
			aggr='mean',
			pos_encoding=self.config.model.tern.image_model.pos_encoding,
			dropout=self.config.model.tern.image_model.dropout
		)

		self.img_proj = nn.Linear(visual_feat_dim, embed_size)
		self.cap_proj = nn.Linear(caption_feat_dim, embed_size)
		self.embed_size = embed_size
		self.shared_transformer = self.config.model.tern.model.shared_transformer

		transformer_layer_1 = nn.TransformerEncoderLayer(
			d_model=embed_size, nhead=4,
			dim_feedforward=2048,
			dropout=dropout, activation='relu'
		)

		self.transformer_encoder_1 = nn.TransformerEncoder(
			transformer_layer_1,
			num_layers=layers
		)

		if not self.shared_transformer:
			transformer_layer_2 = nn.TransformerEncoderLayer(
				d_model=embed_size,
				nhead=4,
				dim_feedforward=2048,
				dropout=dropout,
				activation='relu'
			)
			self.transformer_encoder_2 = nn.TransformerEncoder(
				transformer_layer_2,
				num_layers=layers
			)

		self.text_aggregation = Aggregator(embed_size, aggregation_type=self.config.model.tern.model.text_aggregation)
		self.image_aggregation = Aggregator(embed_size, aggregation_type=self.config.model.tern.model.image_aggregation)
		self.text_aggregation_type = self.config.model.tern.model.text_aggregation
		self.img_aggregation_type = self.config.model.tern.model.image_aggregation

	def forward(self, features, captions, feat_len, cap_len, boxes):
		# process captions by using bert
		full_cap_emb_aggr, c_emb = self.txt_enc(captions, cap_len)  # B x S x cap_dim

		# process image regions using a two-layer transformer
		full_img_emb_aggr, i_emb = self.img_enc(features, feat_len, boxes)  # B x S x vis_dim
		# i_emb = i_emb.permute(1, 0, 2)                             # B x S x vis_dim

		bs = features.shape[0]

		if self.text_aggregation_type is not None:
			c_emb = self.cap_proj(c_emb)

			mask = torch.zeros(bs, max(cap_len)).bool()
			mask = mask.to(features.device)
			for m, c_len in zip(mask, cap_len):
				m[c_len:] = True
			full_cap_emb = self.transformer_encoder_1(
				c_emb.permute(1, 0, 2),
			    src_key_padding_mask=mask)  # S_txt x B x dim
			full_cap_emb_aggr = self.text_aggregation(full_cap_emb, cap_len, mask)
		# else use the embedding output by the txt model
		else:
			full_cap_emb = None

		# forward the regions
		if self.img_aggregation_type is not None:
			i_emb = self.img_proj(i_emb)

			mask = torch.zeros(bs, max(feat_len)).bool()
			mask = mask.to(features.device)
			for m, v_len in zip(mask, feat_len):
				m[v_len:] = True
			if self.shared_transformer:
				full_img_emb = self.transformer_encoder_1(
					i_emb.permute(1, 0, 2),
				    src_key_padding_mask=mask
				)  # S_txt x B x dim
			else:
				full_img_emb = self.transformer_encoder_2(
					i_emb.permute(1, 0, 2),
				                                src_key_padding_mask=mask)  # S_txt x B x dim
			full_img_emb_aggr = self.image_aggregation(full_img_emb, feat_len, mask)
		else:
			full_img_emb = None

		full_cap_emb_aggr = l2norm(full_cap_emb_aggr)
		full_img_emb_aggr = l2norm(full_img_emb_aggr)

		return full_img_emb_aggr, full_cap_emb_aggr, full_img_emb, full_cap_emb


class ImageCaptionEncoder(torch.nn.Module):

	def __init__(self, config):
		"""

		:param config:
		"""

		super().__init__()

		self.config = config

		self.img_txt_enc = JointTextImageTransformerEncoder(config)

		if torch.cuda.is_available():
			logging.info("Using GPU")
			self.to('cuda')
			cudnn.benchmark = True
		else:
			logging.info("Using CPU")
			self.to('cpu')

	def forward_emb(self, images, captions, cap_len, img_len, boxes):
		"""
		Compute the image and caption embeddings
		:param images:
		:param captions:
		:param img_len:
		:param cap_len:
		:param boxes:
		:return:
		"""

		# Forward
		img_emb_aggr, cap_emb_aggr, img_feats, cap_feats = self.img_txt_enc(images, captions, img_len, cap_len, boxes)

		return img_emb_aggr, cap_emb_aggr, img_feats, cap_feats

	def get_parameters(self):
		"""

		:return:
		"""
		lr_multiplier = 1.0 if self.config.model.tern.text_model.fine_tune else 0.0

		ret = []

		params = list(self.img_txt_enc.img_enc.parameters())
		params += list(self.img_txt_enc.img_proj.parameters())
		params += list(self.img_txt_enc.cap_proj.parameters())
		params += list(self.img_txt_enc.transformer_encoder_1.parameters())

		params += list(self.img_txt_enc.image_aggregation.parameters())
		params += list(self.img_txt_enc.text_aggregation.parameters())

		if not self.config.model.tern.model.shared_transformer:
			params += list(self.img_txt_enc.transformer_encoder_2.parameters())

		ret.append(params)

		ret.append(list(self.img_txt_enc.txt_enc.parameters()))

		return ret, lr_multiplier

	def forward(self, images, captions, cap_lengths, boxes=None, ids=None, *args):
		"""
		One training step given images and captions.
		:param images:
		:param captions:
		:param cap_lengths:
		:param boxes:
		:param ids:
		:param args:
		:return:
		"""

		img_lengths = [images.shape[1] for i in range(0, images.shape[0])]

		if torch.cuda.is_available():
			boxes = boxes.cuda()

		# Forward
		img_emb_aggr, cap_emb_aggr, _, _ = self.img_txt_enc(images, captions, img_lengths, cap_lengths, boxes)

		return img_emb_aggr, cap_emb_aggr
