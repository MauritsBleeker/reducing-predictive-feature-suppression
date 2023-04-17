import torch.nn as nn


class TargetDecoder(nn.Module):

	def __init__(self, in_features, hidden_features, reconstruction_dim):
		"""

		:param in_features: size in the input embeddings
		:param hidden_features: size of the hidden state of the network
		:param reconstruction_dim: output dim
		"""
		super(TargetDecoder, self).__init__()

		self.decoder = nn.Sequential(
			nn.Linear(in_features, hidden_features, bias=True),
			nn.ReLU(),
			nn.Linear(hidden_features, hidden_features, bias=True),
			nn.ReLU(),
			nn.Linear(hidden_features, reconstruction_dim, bias=True),
		)

		self.init_weights()

	def forward(self, z_captions):
		"""

		:param z_captions: latent captions
		:return:
		"""
		return self.decoder(z_captions)

	def init_weights(self):
		"""
		Init weights of the network
		:return:
		"""

		nn.init.xavier_uniform_(self.decoder[0].weight)
		nn.init.xavier_uniform_(self.decoder[2].weight)
		nn.init.xavier_uniform_(self.decoder[4].weight)
