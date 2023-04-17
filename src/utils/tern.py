"""
Reference code: https://github.com/mesnico/TERN
"""
import torch
import torch.nn as nn


class GatedAggregation(nn.Module):
	def __init__(self, feat_dim):
		super().__init__()
		self.gate_fn = nn.Sequential(
			nn.Linear(feat_dim, feat_dim),
			nn.ReLU(),
			nn.Linear(feat_dim, 1)
		)
		self.node_fn = nn.Sequential(
			nn.Linear(feat_dim, feat_dim),
			nn.ReLU(),
			nn.Linear(feat_dim, feat_dim)
		)

	def forward(self, x, mask):
		out = x.permute(1, 0, 2)
		gate = self.gate_fn(out)
		gate = gate.masked_fill_(mask.unsqueeze(2), - float('inf'))
		m = torch.sigmoid(gate)  # B x S x 1
		v = self.node_fn(out)  # B x S x dim
		out = torch.bmm(m.permute(0, 2, 1), v)  # B x 1 x dim
		out = out.squeeze(1)  # B x dim
		return out


class Aggregator(nn.Module):

	def __init__(self, embed_size, aggregation_type='sum'):

		super().__init__()
		self.aggregation = aggregation_type
		if self.aggregation == 'gated':
			self.gated_aggr = GatedAggregation(embed_size)
		if self.aggregation == 'gru':
			self.gru_aggr = nn.GRU(embed_size, embed_size, batch_first=True)
		if self.aggregation == 'sum-and-map':

			self.map = nn.Sequential(
				nn.Linear(embed_size, embed_size),
				nn.ReLU(),
				nn.Linear(embed_size, embed_size)
			)

	def forward(self, x, lengths, mask):
		if self.aggregation == 'first':
			out = x[0, :, :]
		elif self.aggregation == 'sum':
			x = x.permute(1, 0, 2)
			for o, c_len in zip(x, lengths):
				o[c_len:] = 0
			out = x.sum(dim=1)
		elif self.aggregation == 'gated':
			out = self.gated_aggr(x, mask)
		elif self.aggregation == 'gru':
			packed_sequence = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=False, enforce_sorted=False)
			_, out = self.gru_aggr(packed_sequence)
			out = out.squeeze(0)
		elif self.aggregation == 'sum-and-map':
			x = x.permute(1, 0, 2)
			for o, c_len in zip(x, lengths):
				o[c_len:] = 0
			out = x.sum(dim=1)
			out = self.map(out)
		else:
			raise ValueError('Final aggregation not defined!')

		return out


class PositionalEncodingImageBoxes(nn.Module):


	def __init__(self, d_model, mode='project-and-sum'):
		"""

		:param d_model:
		:param mode:
		"""

		super().__init__()
		self.mode = mode
		if mode == 'project-and-sum':
			self.map = nn.Linear(5, d_model)
		elif mode == 'concat-and-process':
			self.map = nn.Sequential(
				nn.Linear(d_model + 5, d_model),
				nn.ReLU(),
				nn.Linear(d_model, d_model)
			)

	def forward(self, x, boxes):  # x is seq_len x B x dim
		"""

		:param x:
		:param boxes:
		:return:
		"""
		bs = x.shape[1]
		area = (boxes[:, :, 2] - boxes[:, :, 0]) * (boxes[:, :, 3] - boxes[:, :, 1])
		area = area.unsqueeze(2)
		s_infos = torch.cat([boxes, area], dim=2)
		if self.mode == 'project-and-sum':
			ct = self.map(s_infos).permute(1, 0, 2)    # S x B x dim
			x = x + ct.expand(-1, bs, -1)
		elif self.mode == 'concat-and-process':
			x = torch.cat([x, s_infos.permute(1, 0, 2)], dim=2)
			x = self.map(x)
		return x


def find_nhead(feat_dim, higher=8):
	"""

	:param feat_dim:
	:param higher:
	:return:
	"""
	# find the right n_head value (the highest value lower than 'higher')
	for i in reversed(range(higher + 1)):
		if feat_dim % i == 0:
			return i
	return 1


def l2norm(X):
	"""

	L2-normalize columns of X
	:param X:
	:return:
	"""

	norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
	X = torch.div(X, norm)
	return X
