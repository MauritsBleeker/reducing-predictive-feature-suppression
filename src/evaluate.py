"""
Evaluation class
Reference code:
https://github.com/KunpengLi1994/VSRN/blob/master/evaluation.py
"""
import numpy as np
import fire
from data.dataset import Dataset
from torch.utils.data import DataLoader
from data.dataset import collate_fn
import torch
import logging
import wandb
from models.model import ImageCaptionEncoder
import pickle
import os
from munch import Munch
from utils.evaluation import DCG


class Evaluator(object):
	"""
	Main class to run evaluation on the validation or test set.

	"""
	def __init__(self, config, split, model, cross_dataset=False):
		"""

		:param config:
		:param split:
		:param model:
		:param cross_dataset:
		"""

		self.config = config

		assert split == 'val' or split == 'test'

		self.split = split

		self.model = model

		pickle_file = self.config.dataset.val_pickle if split == 'val' else self.config.dataset.test_pickle

		logging.info("Loading {} for the evaluation set".format(pickle_file))

		self.dataset = Dataset(
			pickle_file=pickle_file,
			config=self.config,
			split=split,
			cross_dataset=cross_dataset
		)

		self.dataloader = DataLoader(
			self.dataset,
			shuffle=False,
			batch_size=self.config.dataloader.eval_batch_size,
			num_workers=self.config.dataloader.num_workers,
			collate_fn=collate_fn,
			pin_memory=True
		)

		self.cross_dataset = cross_dataset

		if self.config.dataset.dataset_name == 'coco':
			self.ndcg_val_scorer = DCG(
				config, len(self.dataset.caption_ids), split, rank=25, relevance_methods=['rougeL', 'spice']
			)

	@torch.no_grad()
	def encode_data(self):
		"""
		Encode the test data
		:return: image_representations, caption_representations, img_ids, cap_ids
		"""

		caption_representations = np.zeros((len(self.dataset.caption_ids), self.config.model.embed_dim))
		image_representations = np.zeros((len(self.dataset.caption_ids), self.config.model.embed_dim))

		img_ids = np.zeros((len(self.dataset.caption_ids), 1))
		cap_ids = np.zeros((len(self.dataset.caption_ids), 1))

		for i, (images, tokens, cap_lengths, targets, caption_ids, image_ids, idx, mask, img_boxes) in enumerate(self.dataloader):

			z_images, z_captions, _ = self.model(
				images.to(self.model.img_encoder_device),
				tokens.to(self.model.cap_encoder_device),
				cap_lengths=cap_lengths, 
				boxes=img_boxes,
			)

			caption_representations[idx, :] = z_captions.cpu().numpy().copy()
			image_representations[idx, :] = z_images.cpu().numpy().copy()
			img_ids[idx, :] = np.array(image_ids).reshape(-1, 1)
			cap_ids[idx, :] = np.array(caption_ids).reshape(-1, 1)

		return image_representations, caption_representations, img_ids, cap_ids

	def i2t(self, image_representations, caption_representations):
		"""
		Perform image two text retrieval over the entire val/test set
		:param image_representations:
		:param caption_representations:
		:return: r1, r5, r10, medr, meanr, r_precision
		"""

		n_images = len(self.dataset) // self.config.dataset.captions_per_image

		ranks = np.zeros(n_images)
		top1 = np.zeros(n_images)
		r_precision = []

		for index in range(0, n_images):
			z_img = image_representations[self.config.dataset.captions_per_image * index].reshape(1, image_representations.shape[1])

			sim = np.dot(z_img, caption_representations.T).flatten()
			ranking = np.argsort(sim)[::-1]

			rank = 1e20
			for i in range(self.config.dataset.captions_per_image * index, self.config.dataset.captions_per_image * index + self.config.dataset.captions_per_image, 1):
				tmp = np.where(ranking == i)[0][0]
				if tmp < rank:
					rank = tmp

			r_precision .append(
				((ranking[:self.config.dataset.captions_per_image] >= self.config.dataset.captions_per_image * index) &
				(ranking[:self.config.dataset.captions_per_image] < self.config.dataset.captions_per_image * index + self.config.dataset.captions_per_image)).sum() / self.config.dataset.captions_per_image
			)

			ranks[index] = rank
			top1[index] = ranking[0]

		r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
		r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
		r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

		medr = np.floor(np.median(ranks)) + 1
		meanr = ranks.mean() + 1
		r_precision = np.mean(r_precision)

		return r1, r5, r10, medr, meanr, r_precision

	def t2i(self, image_representations, caption_representations):
		"""
		Perform text two image retrieval over the entire val/test set
		:param image_representations: latent images
		:param caption_representations: latent captions
		:return: r1, r5, r10, medr, meanr,  mean_rougel_ndcg, mean_spice_ndcg
		"""

		npts = image_representations.shape[0] // self.config.dataset.captions_per_image

		ims = np.array([image_representations[i] for i in range(0, len(image_representations), self.config.dataset.captions_per_image)])

		ranks = np.zeros(self.config.dataset.captions_per_image * npts)
		top1 = np.zeros(self.config.dataset.captions_per_image * npts)

		rougel_ndcgs = np.zeros(self.config.dataset.captions_per_image * npts)
		spice_ndcgs = np.zeros(self.config.dataset.captions_per_image * npts)

		for index in range(npts):

			# Get query captions
			queries = caption_representations[
				self.config.dataset.captions_per_image * index:self.config.dataset.captions_per_image * index + self.config.dataset.captions_per_image
			]

			d = np.dot(queries, ims.T)

			inds = np.zeros(d.shape)
			
			for i in range(len(inds)):
				inds[i] = np.argsort(d[i])[::-1]
				ranks[self.config.dataset.captions_per_image * index + i] = np.where(inds[i] == index)[0][0]
				top1[self.config.dataset.captions_per_image * index + i] = inds[i][0]

				if self.config.dataset.dataset_name == 'coco':

					rougel_ndcgs[5 * index + i], spice_ndcgs[5 * index + i] = self.ndcg_val_scorer.compute_ndcg(
							npts, 5 * index + i, inds[i].astype(int), fold_index=0, retrieval='image'
						).values()

		# Compute metrics

		r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
		r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
		r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

		medr = np.floor(np.median(ranks)) + 1
		meanr = ranks.mean() + 1

		mean_rougel_ndcg = np.mean(rougel_ndcgs)
		mean_spice_ndcg = np.mean(spice_ndcgs)

		return r1, r5, r10, medr, meanr,  mean_rougel_ndcg, mean_spice_ndcg

	def evaluate(self, step=None):
		"""
		Evaluate a model
		:param step: current iteration/training set
		:return: rsum: sum of all recall scores
		"""

		self.model.eval()

		image_representations, caption_representations, img_ids, cap_ids = self.encode_data()

		i2t_r1, i2t_r5, i2t_r10, i2t_medr, i2t_meanr, r_precision = self.i2t(image_representations, caption_representations)

		t2i_r1, t2i_r5, t2i_r10, t2i_medr, t2i_meanr,  mean_rougel_ndcg, mean_spice_ndcg = self.t2i(image_representations, caption_representations)

		rsum = i2t_r1 + i2t_r5 + i2t_r10 + t2i_r1 + t2i_r5 + t2i_r10

		self.log_results(
			i2t_r1, i2t_r5, i2t_r10, i2t_medr, i2t_meanr,
			t2i_r1, t2i_r5, t2i_r10, t2i_medr, t2i_meanr, rsum, step, r_precision,
			mean_rougel_ndcg, mean_spice_ndcg
		)

		if self.split == 'test' and self.config.dataset.dataset_name == 'coco' and not self.cross_dataset:
			self.cxc_i2t(image_representations, caption_representations, img_ids, cap_ids)
			self.cxc_t2i(image_representations, caption_representations, img_ids, cap_ids)

		return rsum

	def cxc_i2t(self, image_representations, caption_representations, img_ids, cap_ids):
		"""
		i2t for CxC dataset
		:param image_representations: latents of the images
		:param caption_representations: latents of the captions
		:param img_ids: image_ids
		:param cap_ids: caption_ids
		:return:
		"""

		n_images = len(self.dataset) // self.config.dataset.captions_per_image

		r1 = r5 = r10 = 0
		r_precision = []

		for index in range(0, n_images):
			z_img = image_representations[self.config.dataset.captions_per_image * index].reshape(1, image_representations.shape[1])

			img_id = int(img_ids[index * self.config.dataset.captions_per_image][0])

			sim = np.dot(z_img, caption_representations.T).flatten()
			ranking = np.argsort(sim)[::-1]

			r1 += 1 if self.dataset.dataset['images'][img_id]['i2c']['caption_ids'] in cap_ids[
				ranking[:1]] else 0
			r5 += 1 if self.dataset.dataset['images'][img_id]['i2c']['caption_ids'] in cap_ids[
				ranking[:5]] else 0
			r10 += 1 if self.dataset.dataset['images'][img_id]['i2c']['caption_ids'] in cap_ids[
				ranking[:10]] else 0

			n_positive = len(self.dataset.dataset['images'][img_id]['i2c']['caption_ids'])

			r_precision.append(
				len(
					set(self.dataset.dataset['images'][img_id]['i2c']['caption_ids'])
					& set(cap_ids[ranking[:n_positive]].reshape(-1).tolist())
				) / n_positive
			)

		r1 = 100 * (r1 / n_images)
		r5 = 100 * (r5 / n_images)
		r10 = 100 * (r10 / n_images)
		r_precision = np.mean(r_precision)

		logging.info("CxC Image to text: %.1f, %.1f, %.1f" % (r1, r5, r10))
		logging.info("CxC Image to text r-precision: %.2f" % (r_precision))

	def cxc_t2i(self, image_representations, caption_representations, img_ids, cap_ids):
		"""
		t2i for CxC dataset
		:param image_representations: latents of the images
		:param caption_representations: latents of the captions
		:param img_ids: image_ids
		:param cap_ids: caption_ids
		:return:
		"""

		npts = image_representations.shape[0] // self.config.dataset.captions_per_image

		ims = np.array([image_representations[i] for i in range(0, len(image_representations), self.config.dataset.captions_per_image)])

		n_caption = len(caption_representations)

		r1 = r5 = r10 = 0

		r_precision = []

		for index in range(npts):

			# Get query captions
			queries = caption_representations[self.config.dataset.captions_per_image * index:self.config.dataset.captions_per_image * index + self.config.dataset.captions_per_image]

			sim = np.dot(queries, ims.T)

			inds = np.zeros(sim.shape)
			for i in range(len(inds)):
				ranking = np.argsort(sim[i])[::-1]

				caption_id = int(cap_ids[self.config.dataset.captions_per_image * index + i][0])
				r1 += 1 if self.dataset.dataset['captions'][caption_id]['c2i']['img_ids'] in img_ids[::5][
					ranking[:1]] else 0
				r5 += 1 if self.dataset.dataset['captions'][caption_id]['c2i']['img_ids'] in img_ids[::5][
					ranking[:5]] else 0
				r10 += 1 if self.dataset.dataset['captions'][caption_id]['c2i']['img_ids'] in img_ids[::5][
					ranking[:10]] else 0

				n_positive = len(self.dataset.dataset['captions'][caption_id]['c2i']['img_ids'])

				r_precision.append(
					len(
						set(self.dataset.dataset['captions'][caption_id]['c2i']['img_ids'])
						&
						set(img_ids[::self.config.dataset.captions_per_image][ranking[:n_positive]].reshape(-1).tolist())
					) / n_positive
				)

		r1 = 100 * (r1 / n_caption)
		r5 = 100 * (r5 / n_caption)
		r10 = 100 * (r10 / n_caption)
		r_precision = np.mean(r_precision)

		logging.info("CxC Text to image: %.1f, %.1f, %.1f" % (r1, r5, r10))
		logging.info("CxC Text to image r_precision: %.2f" % (r_precision))

	def log_results(self, i2t_r1, i2t_r5, i2t_r10, i2t_medr, i2t_meanr, t2i_r1, t2i_r5, t2i_r10, t2i_medr, t2i_meanr, rsum, step, r_precision, mean_rougel_ndcg, mean_spice_ndcg):

		if self.split == 'val':
			wandb.log({
				'i2t_r1': i2t_r1,
				'i2t_r5': i2t_r5,
				'i2t_r10': i2t_r10,
				'i2t_medr': i2t_medr,
				'i2t_meanr': i2t_meanr,
				't2i_r1': t2i_r1,
				't2i_r5': t2i_r5,
				't2i_r10': t2i_r10,
				't2i_medr': t2i_medr,
				't2i_meanr': t2i_meanr,
				'rsum': rsum,
				'r_precision': r_precision
			}, step=step)

		logging.info("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" %(i2t_r1, i2t_r5, i2t_r10, i2t_medr, i2t_meanr))
		logging.info("Image to text r-precision: %.2f" % (r_precision))
		logging.info("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" % (t2i_r1, t2i_r5, t2i_r10, t2i_medr, t2i_meanr))

		if self.config.dataset.dataset_name == 'coco':
			logging.info("Mean Rougel ndcg: %.4f" % (mean_rougel_ndcg))
			logging.info("Mean Spice ndcg: %.4f" % (mean_spice_ndcg))

			if self.split == 'val':
				wandb.log({
					'mean_rougel_ndcg': mean_rougel_ndcg,
					'mean_spice_ndcg': mean_spice_ndcg
				}, step=step)

def main(path_to_model, split='test', data_root=None, cross_dataset=False):
	"""

	:param path_to_model: path to model checkpoint
	:param split: test/validate split
	:param data_root: root folder of the data
	:param cross_dataset: evaluate with mscoco while the model has been trained on F30k, or the other way around.
	:return:
	"""

	checkpoint = torch.load(path_to_model, map_location='cuda' if torch.cuda.is_available() else 'cpu')

	config = Munch.fromDict(checkpoint['config'])

	if data_root:
		config.dataset.root = data_root

	logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

	if 'vocab' in checkpoint:
		vocab = checkpoint['vocab']
	else:
		vocab = pickle.load(open(os.path.join(config.dataset.root, config.dataset.dataset_name, 'vocab', config.dataset.vocab_file),'rb'))

	model = ImageCaptionEncoder(
		config=config,
		vocab=vocab,
		init_weights=False
	)

	model.load_checkpoint(checkpoint, strict=False)

	evaluator = Evaluator(config=config, split=split, model=model, cross_dataset=cross_dataset)

	evaluator.evaluate()


if __name__ == '__main__':
	fire.Fire(main)
