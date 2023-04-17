"""
Image Encoder of the VSRN model.
Slightly refactored.
Reference code:
https://github.com/KunpengLi1994/VSRN/blob/master/model.py
"""
import torch.nn as nn
from libs.GCN_lib.Rs_GCN import Rs_GCN
from utils.utils import l2_normalize
import numpy as np


class ImageEncoder(nn.Module):

	def __init__(self, config):
		super(ImageEncoder, self).__init__()

		self.config = config

		self.embed_dim = self.config.model.vsrn.embed_dim
		self.no_imgnorm = self.config.model.vsrn.no_imgnorm
		self.dataset_name = self.config.dataset.dataset_name
		self.img_dim = self.config.model.vsrn.image_encoder.img_dim

		self.fc = nn.Linear(self.img_dim, self.embed_dim)

		self.init_weights()

		# GSR
		self.img_rnn = nn.GRU(self.embed_dim, self.embed_dim, 1, batch_first=True)

		# GCN reasoning
		self.Rs_GCN_1 = Rs_GCN(in_channels=self.embed_dim, inter_channels=self.embed_dim)
		self.Rs_GCN_2 = Rs_GCN(in_channels=self.embed_dim, inter_channels=self.embed_dim)
		self.Rs_GCN_3 = Rs_GCN(in_channels=self.embed_dim, inter_channels=self.embed_dim)
		self.Rs_GCN_4 = Rs_GCN(in_channels=self.embed_dim, inter_channels=self.embed_dim)

		self.bn = nn.BatchNorm1d(self.embed_dim)

	def init_weights(self):
		"""
		Xavier initialization for the fully connected layer
		"""
		r = np.sqrt(6.) / np.sqrt(self.fc.in_features + self.fc.out_features)
		self.fc.weight.data.uniform_(-r, r)
		self.fc.bias.data.fill_(0)

	def forward(self, images):
		"""Extract image feature vectors."""

		fc_img_emd = self.fc(images)
		if self.dataset_name != 'f30k':
			fc_img_emd = l2_normalize(fc_img_emd)

		# GCN reasoning
		# -> B,D,N
		gcn_img_emd = fc_img_emd.permute(0, 2, 1)
		gcn_img_emd = self.Rs_GCN_1(gcn_img_emd)
		gcn_img_emd = self.Rs_GCN_2(gcn_img_emd)
		gcn_img_emd = self.Rs_GCN_3(gcn_img_emd)
		gcn_img_emd = self.Rs_GCN_4(gcn_img_emd)
		# -> B,N,D
		gcn_img_emd = gcn_img_emd.permute(0, 2, 1)

		gcn_img_emd = l2_normalize(gcn_img_emd)

		rnn_img, hidden_state = self.img_rnn(gcn_img_emd)

		# features = torch.mean(rnn_img,dim=1)
		out = hidden_state[0]

		#if self.dataset_name == 'f30k':
		out = self.bn(out)

		# normalize in the joint embedding space

		out = l2_normalize(out)

		return out
