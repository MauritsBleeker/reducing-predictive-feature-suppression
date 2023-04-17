from torch.utils.data import Dataset
import os
import pickle
from PIL import Image
from six import BytesIO as IO
from nltk.tokenize import word_tokenize
import random
import torch
from utils.transform import imagenet_transform
import logging
import numpy as np
from transformers import BertTokenizer


class Dataset(Dataset):

	def __init__(self, pickle_file, split, config, cross_dataset=False):
		"""

		:param pickle_file:
		:param split:
		:param config:
		:param cross_dataset:
		"""

		self.config = config
		self.split = split

		self.dataset_name = self.config.dataset.dataset_name

		if cross_dataset:
			self.dataset_name = 'coco' if self.config.dataset.dataset_name == 'f30k' else 'f30k'

		self.dataset = pickle.load(open(os.path.join(self.config.dataset.root, self.dataset_name, pickle_file),'rb'))

		self.image_transform = imagenet_transform(
			random_resize_crop=self.split == 'train',
			random_erasing_prob=self.config.dataloader.random_erasing_prob if self.split == 'train' else 0,
		)

		self.caption_ids = list(self.dataset['captions'])
		self.image_ids = list(self.dataset['images'])

		logging.info("Loading vocab: {}".format(self.config.dataset.vocab_file))
		# when evaluating cross dataset, we still need to load the vocab of the original dataset, since otherwise the embeddings wont match
		self.vocab = pickle.load(open(os.path.join(self.config.dataset.root, self.config.dataset.dataset_name, 'vocab', self.config.dataset.vocab_file),'rb'))

		logging.info(f'Loaded {self.dataset_name} Split: {split},  n_images {len(self.image_ids)} n_captions {len(self.caption_ids)}')

		self.target_key = 'target' if 'target_key' not in self.config.dataloader else self.config.dataloader.target_key

		self.use_vsrn = False
		self.use_tern = False

		if 'vsrn' in self.config.model and self.config.model.vsrn.use_model:
			self.use_vsrn = self.config.model.vsrn.use_model

			self.data_ids = open(
				os.path.join(
					self.config.dataset.root,
					self.config.dataset.dataset_name + '_precomp',
					'%s_ids.txt' % (self.split if split == 'val' or self.dataset_name == 'f30k' else self.split + 'all'))
			).read().split('\n')

			self.precomp_images = np.load(
				os.path.join(
					self.config.dataset.root, self.config.dataset.dataset_name + '_precomp', '%s_ims.npy' % (self.split if split == 'val' or self.dataset_name == 'f30k' else self.split + 'all'))
			)

			logging.info("Loaded pre-compted images for VSRN.")

			if self.config.dataset.dataset_name == 'coco':
				self._init_coco_vsrn()

		elif 'TERN' in self.config.model and self.config.model.tern.use_model:
			self.use_tern = self.config.model.tern.use_model
			self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

		assert not (self.use_vsrn and self.use_tern)

	def __len__(self):

		return len(self.caption_ids)

	def __getitem__(self, idx):
		"""

		:param idx:
		:return:
		"""

		caption_id = self.caption_ids[idx]
		caption = self.dataset['captions'][caption_id]
		image_id = caption['imgid']
		cocoid = self.dataset['images'][image_id]['cocoid'] if self.dataset_name == 'coco' else None

		image, boxes = self.load_image(idx, image_id, cocoid=cocoid)

		tokens = self.tokenize(
			caption['caption'],
			self.vocab,
			caption_drop_prob=self.config.dataloader.caption_drop_prob if self.split == 'train' else 0
		)

		target = torch.Tensor(caption[self.target_key])

		return image, tokens, caption_id, image_id, idx, target, boxes

	def _init_coco_vsrn(self):
		"""
		There is a slight mismatch between the precomputed features dataset by VSRN and the one we used.
		Moreover, the image features for the val and test set are duplicated, therefore we need to do some bookkeeping
		to match these feature with our dataset.
		:return: None
		"""
		if self.config.dataset.dataset_name == 'coco':

			self.cocoid_to_idx = {}
			for i, coco_id in enumerate(self.data_ids):
				if coco_id != '':
					self.cocoid_to_idx[int(coco_id)] = i

			if self.split == 'val':

				coco_id_to_img = {}

				for key, image in self.dataset['images'].items():
					coco_id_to_img[image['cocoid']] = key

				caption_ids = []

				for data_id in self.data_ids[::self.config.dataset.captions_per_image]:
					if data_id != '':
						caption_ids.extend(
							self.dataset['images'][coco_id_to_img[int(data_id)]]['caption_ids'][:self.config.dataset.captions_per_image]
						)

				self.caption_ids = caption_ids

	def load_image(self, idx, image_id, cocoid=None):
		"""
		:param idx: index of the sampled datapoint
		:param  image_id: id of the image in the dataset
		:return:
		"""

		img_boxes =  None
		if self.use_vsrn:
			if self.config.dataset.dataset_name == 'f30k':
				img_idx = idx
				if self.split == 'train':
					img_idx = idx // self.config.dataset.captions_per_image
				assert int(self.data_ids[img_idx]) == image_id
			elif self.config.dataset.dataset_name == 'coco':
				img_idx = self.cocoid_to_idx[self.dataset['images'][image_id]['cocoid']]
			else:
				raise Exception("Unkown dataset")

			image = torch.Tensor(self.precomp_images[img_idx])

		elif self.use_tern:

			image = torch.Tensor(self.dataset['images'][image_id]['image'])
			img_boxes = torch.Tensor(self.dataset['images'][image_id]['img_boxes'])

			img_size = np.array([
					self.dataset['images'][image_id]['size']['width'],
					self.dataset['images'][image_id]['size']['height']
				]
			)
			img_boxes = img_boxes / np.tile(img_size, 2)

		else:

			image = Image.open(IO(self.dataset['images'][image_id]['image'])).convert('RGB')
			image = self.image_transform(image)

		return image, img_boxes

	def tokenize(self, sentence, vocab, caption_drop_prob=0):
		"""
		nltk word_tokenize for caption transform.
		:param sentence:
		:param vocab:
		:param caption_drop_prob:
		:return:
		"""
		if self.use_tern:
			tokenized_sentence = self.bert_tokenizer(sentence)['input_ids']

		else:

			tokens = word_tokenize(str(sentence).lower())
			tokenized_sentence = list()
			tokenized_sentence.append(vocab('<start>'))

			if caption_drop_prob > 0:
				unk = vocab('<unk>')
				tokenized = [vocab(token) if random.random() > caption_drop_prob else unk for token in tokens]
			else:
				tokenized = [vocab(token) for token in tokens]

			if caption_drop_prob:

				N = int(len(tokenized) * caption_drop_prob)

				for _ in range(N):
					tokenized.pop(random.randrange(len(tokenized)))

			tokenized_sentence.extend(tokenized)
			tokenized_sentence.append(vocab('<end>'))

		return torch.Tensor(tokenized_sentence)


def collate_fn(data):
	"""

	:param data:
	:return:
	"""

	# Sort a data list by sentence length
	data.sort(key=lambda x: len(x[1]), reverse=True)
	images, captions, caption_ids, image_ids, idx, targets, img_boxes = zip(*data)

	# Merge images (convert tuple of 3D tensor to 4D tensor)
	preextracted_images = not (images[0].shape[0] == 3) and img_boxes[0] is not None

	if preextracted_images:
		feat_lengths = [f.shape[0] + 1 for f in images]  # +1 because the first region feature is reserved as CLS

		feat_dim = images[0].shape[1]
		img_features = torch.zeros(len(images), max(feat_lengths), feat_dim)

		for i, img in enumerate(images):
			end = feat_lengths[i]
			img_features[i, 1:end] = img

		images = img_features

		box_lengths = [b.shape[0] + 1 for b in img_boxes]  # +1 because the first region feature is reserved as CLS
		assert box_lengths == feat_lengths
		out_boxes = torch.zeros(len(img_boxes), max(box_lengths), 4)
		for i, box in enumerate(img_boxes):
			end = box_lengths[i]
			out_boxes[i, 1:end] = box
		img_boxes = out_boxes

	else:
		images = torch.stack(images, 0)
		img_boxes = None

	targets = torch.stack(targets, 0)
	# Merge sentences (convert tuple of 1D tensor to 2D tensor)
	cap_lengths = [len(cap) for cap in captions]
	tokens = torch.zeros(len(captions), max(cap_lengths)).long()

	mask = torch.zeros(len(captions), max(cap_lengths)).long()

	for i, cap in enumerate(captions):
		end = cap_lengths[i]
		tokens[i, :end] = cap[:end]
		mask[i, :end] = 1

	cap_lengths = torch.Tensor(cap_lengths).long()

	return images, tokens, cap_lengths, targets, caption_ids, image_ids, idx, mask, img_boxes
