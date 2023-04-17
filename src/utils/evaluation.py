"""
Reference code: https://github.com/mesnico/TERN/blob/master/evaluate_utils/dcg.py
"""
import os
import numpy as np


class DCG:

	def __init__(self, config, n_queries, split, rank=25, relevance_methods=['rougeL']):

		self.rank = rank
		self.config = config
		self.relevance_methods = relevance_methods

		relevance_dir = os.path.join(config.dataset.root, config.dataset.dataset_name, 'relevances')
		relevance_filenames = [os.path.join(relevance_dir, '{}-{}-{}.npy'.format(config.dataset.dataset_name,
																			   split, m))
							   for m in relevance_methods]
		self.relevances = [np.memmap(f, dtype=np.float32, mode='r') for f in relevance_filenames]
		for r in self.relevances:
			r.shape = (n_queries, -1)

	def compute_ndcg(self, npts, query_id, sorted_indexes, fold_index=0, retrieval='image'):
		"""

		:param npts:
		:param query_id:
		:param sorted_indexes:
		:param fold_index:
		:param retrieval:
		:return:
		"""
		sorted_indexes = sorted_indexes[:self.rank]
		# npts = self.relevances[0].shape[1] // 5
		if retrieval == 'image':
			query_base = npts * self.config.dataset.captions_per_image * fold_index
			# sorted_indexes += npts * fold_index
			relevances = [r[query_base + query_id, fold_index * npts : (fold_index + 1) * npts] for r in self.relevances]
		elif retrieval == 'sentence':
			query_base = npts * fold_index
			# sorted_indexes += npts * 5 * fold_index
			relevances = [r[fold_index * npts * self.config.dataset.captions_per_image : (fold_index + 1) * npts * self.config.dataset.captions_per_image, query_base + query_id] for r in self.relevances]

		ndcg_scores = [self.ndcg_from_ranking(r, sorted_indexes) for r in relevances]
		out = {k: v for k, v in zip(self.relevance_methods, ndcg_scores)}
		return out

	@staticmethod
	def ranking_precision_score(y_true, y_score, k=10):
		"""
		Precision at rank k
		Parameters
		----------
		y_true : array-like, shape = [n_samples]
			Ground truth (true relevance labels).
		y_score : array-like, shape = [n_samples]
			Predicted scores.
		k : int
			Rank.
		Returns
		-------
		precision @k : float
		"""
		unique_y = np.unique(y_true)

		if len(unique_y) > 2:
			raise ValueError("Only supported for two relevance levels.")

		pos_label = unique_y[1]
		n_pos = np.sum(y_true == pos_label)

		order = np.argsort(y_score)[::-1]
		y_true = np.take(y_true, order[:k])
		n_relevant = np.sum(y_true == pos_label)

		# Divide by min(n_pos, k) such that the best achievable score is always 1.0.
		return float(n_relevant) / min(n_pos, k)

	@staticmethod
	def average_precision_score(y_true, y_score, k=10):
		"""
		Average precision at rank k
		Parameters
		----------
		y_true : array-like, shape = [n_samples]
			Ground truth (true relevance labels).
		y_score : array-like, shape = [n_samples]
			Predicted scores.
		k : int
			Rank.
		Returns
		-------
		average precision @k : float
		"""
		unique_y = np.unique(y_true)

		if len(unique_y) > 2:
			raise ValueError("Only supported for two relevance levels.")

		pos_label = unique_y[1]
		n_pos = np.sum(y_true == pos_label)

		order = np.argsort(y_score)[::-1][:min(n_pos, k)]
		y_true = np.asarray(y_true)[order]

		score = 0
		for i in range(len(y_true)):
			if y_true[i] == pos_label:
				# Compute precision up to document i
				# i.e, percentage of relevant documents up to document i.
				prec = 0
				for j in range(0, i + 1):
					if y_true[j] == pos_label:
						prec += 1.0
				prec /= (i + 1.0)
				score += prec

		if n_pos == 0:
			return 0

		return score / n_pos

	@staticmethod
	def dcg_score(y_true, y_score, k=10, gains="exponential"):
		"""
		Discounted cumulative gain (DCG) at rank k
		Parameters
		----------
		y_true : array-like, shape = [n_samples]
			Ground truth (true relevance labels).
		y_score : array-like, shape = [n_samples]
			Predicted scores.
		k : int
			Rank.
		gains : str
			Whether gains should be "exponential" (default) or "linear".
		Returns
		-------
		DCG @k : float
		"""
		order = np.argsort(y_score)[::-1]
		y_true = np.take(y_true, order[:k])

		if gains == "exponential":
			gains = 2 ** y_true - 1
		elif gains == "linear":
			gains = y_true
		else:
			raise ValueError("Invalid gains option.")

		# highest rank is 1 so +2 instead of +1
		discounts = np.log2(np.arange(len(y_true)) + 2)
		return np.sum(gains / discounts)

	def ndcg_score(self, y_true, y_score, k=10, gains="exponential"):
		"""
		Normalized discounted cumulative gain (NDCG) at rank k
		Parameters
		----------
		y_true : array-like, shape = [n_samples]
			Ground truth (true relevance labels).
		y_score : array-like, shape = [n_samples]
			Predicted scores.
		k : int
			Rank.
		gains : str
			Whether gains should be "exponential" (default) or "linear".
		Returns
		-------
		NDCG @k : float
		"""
		best = self.dcg_score(y_true, y_true, k, gains)
		actual = self.dcg_score(y_true, y_score, k, gains)
		return actual / best

	@staticmethod
	def dcg_from_ranking(y_true, ranking):
		"""
		Discounted cumulative gain (DCG) at rank k
		Parameters
		----------
		y_true : array-like, shape = [n_samples]
			Ground truth (true relevance labels).
		ranking : array-like, shape = [k]
			Document indices, i.e.,
				ranking[0] is the index of top-ranked document,
				ranking[1] is the index of second-ranked document,
				...
		k : int
			Rank.
		Returns
		-------
		DCG @k : float
		"""
		y_true = np.asarray(y_true)
		ranking = np.asarray(ranking)
		rel = y_true[ranking]
		gains = 2 ** rel - 1
		discounts = np.log2(np.arange(len(ranking)) + 2)
		return np.sum(gains / discounts)

	def ndcg_from_ranking(self, y_true, ranking):
		"""
		Normalized discounted cumulative gain (NDCG) at rank k
		Parameters
		----------
		y_true : array-like, shape = [n_samples]
			Ground truth (true relevance labels).
		ranking : array-like, shape = [k]
			Document indices, i.e.,
				ranking[0] is the index of top-ranked document,
				ranking[1] is the index of second-ranked document,
				...
		k : int
			Rank.
		Returns
		-------
		NDCG @k : float
		"""
		k = len(ranking)
		best_ranking = np.argsort(y_true)[::-1]
		best = self.dcg_from_ranking(y_true, best_ranking[:k])
		if best == 0:
			return 0
		return self.dcg_from_ranking(y_true, ranking) / best
