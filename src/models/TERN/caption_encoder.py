"""
Reference code:
"""
import torch
from torch import nn as nn
from transformers import BertTokenizer, BertModel, BertConfig


class EncoderTextBERT(nn.Module):

	def __init__(self, config, mean=True, post_transformer_layers=0):
		super().__init__()

		self.config = config

		self.preextracted = self.config.model.tern.text_model.pre_extracted

		bert_config = BertConfig.from_pretrained(
			self.config.model.tern.text_model.pretrain,
			output_hidden_states=True,
			num_hidden_layers=self.config.model.tern.text_model.extraction_hidden_layer
		)

		bert_model = BertModel.from_pretrained(
			self.config.model.tern.text_model.pretrain,
			config=bert_config,
			cache_dir=self.config.experiment.cache_dir
		)

		self.vocab_size = bert_model.config.vocab_size
		self.hidden_layer = self.config.model.tern.text_model.extraction_hidden_layer

		if not self.preextracted:

			self.tokenizer = BertTokenizer.from_pretrained(self.config.model.tern.text_model.pretrain)
			self.bert_model = bert_model
			self.word_embeddings = self.bert_model.get_input_embeddings()

		if post_transformer_layers > 0:

			transformer_layer = nn.TransformerEncoderLayer(
				d_model=self.config.model.tern.text_model.word_dim,
				nhead=4,
				dim_feedforward=2048,
				dropout=self.config.model.tern.text_model.dropout, activation='relu'
			)

			self.transformer_encoder = nn.TransformerEncoder(
				transformer_layer,
				num_layers=post_transformer_layers
			)

		self.post_transformer_layers = post_transformer_layers

		self.map = nn.Linear(self.config.model.tern.text_model.word_dim, self.config.model.tern.model.embed_size)

		self.mean = mean

	def forward(self, captions, lengths):
		"""

		:param captions:
		:param lengths:
		:return:
		"""
		if not self.preextracted or self.post_transformer_layers > 0:
			max_len = max(lengths)
			attention_mask = torch.ones(captions.shape[0], max_len)
			for e, l in zip(attention_mask, lengths):
				e[l:] = 0
			attention_mask = attention_mask.to(captions.device)

		if self.preextracted:
			outputs = captions
		else:
			outputs = self.bert_model(captions, attention_mask=attention_mask)
			outputs = outputs[2][-1]

		if self.post_transformer_layers > 0:
			outputs = outputs.permute(1, 0, 2)
			outputs = self.transformer_encoder(outputs, src_key_padding_mask=(attention_mask - 1).bool())
			outputs = outputs.permute(1, 0, 2)
		if self.mean:
			captions = outputs.mean(dim=1)
		else:
			captions = outputs[:, 0, :]     # from the last layer take only the first word

		out = self.map(captions)

		return out, outputs

	def get_finetuning_params(self):
		"""

		:param self:
		:return:
		"""
		return list(self.bert_model.parameters())
