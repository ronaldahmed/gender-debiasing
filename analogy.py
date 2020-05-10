import numpy as np
import numpy.linalg as la
import yaml
import argparse
import fnmatch
import os
import gensim.downloader as api
from tqdm import tqdm
import itertools
import ray


ray.init()

models = {
	"w2v": "word2vec-google-news-300",
	"glove_wiki": "glove-wiki-gigaword-300",
	"glove_twitter": "glove-twitter-200",
}


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--model', type=str, default='w2v', help="Select embedding model {w2v, glove_wiki, glove_twitter}")
	parser.add_argument('--show_ranking', type=str, default='', help="Path to YAML file with ranking results")
	parser.add_argument('--top', type=int, default=20, help="Number of results to show")
	parser.add_argument('--filter', type=bool, default=True, help="If filter True, only show pairs with different words. Default: True")
	parser.add_argument('--embeds_path', type=str, default='', help="Path to YAML file with dictionary of embeddings word -> vector")
	parser.add_argument('--x', type=str, help="Word")
	parser.add_argument('--y', type=str, help="Word")
	parser.add_argument('--vocab', type=str, default='', help="Path to Vocab YAML file, can be set or dict of word->index")
	return parser.parse_args()


@ray.remote
def compute_score(u, w1, w2, embeds):
	v = embeds[w1] - embeds[w2]
	score = np.dot(u, v) / (la.norm(u) * la.norm(v))
	return w1, w2, score


# @ray.remote
def is_neighbor(u, v, embeds, delta=1):
	d = embeds[u] - embeds[v]
	if delta >= la.norm(d) >= 1e-8:
		return True
	return False


@ray.remote
def compute_neighbors(w, embeds, delta=1):
	N = [token for token in embeds if is_neighbor(w, token, embeds, delta=delta)]
	# N = ray.get(N)
	N.append(w)
	print(f"Neighbor size {w}: {len(N)}")
	return N


def compute_score_analogy_pairs(x, y, embeds, vocab=None, delta=1):
	"""
	Compute scores for analogy pairs (a, b) such that Norm(a-b) <= delta
	:param x: Index of word
	:param y: Index of word
	:param embeds: Dictionary of embeddings word -> embedding
	:param vocab: set of words or dict of word -> index to find the analogy pairs
	:param delta: threshold for semantic similarity of analogy pairs (delta = 1)
	:return: List of (a,b,score) tuples ranked by their score in non-increasing order, i.e., ranking[0] has higher score
	"""
	print("Computing scores...")
	ranking = []
	normed_vecs = {}
	if vocab is None:
		vocab = embeds.keys()

	for token in vocab:
		try:
			emb = embeds[token]
			normed_vecs[token] = emb / la.norm(emb)
		except:
			continue
	for token in [x, y]:
		try:

			emb = embeds[token]
			normed_vecs[token] = emb / la.norm(emb)
		except:
			assert 0, f"Embedding of {token} not found"

	print(f"> Total # of words in vocab with embedding {len(normed_vecs)}")
	tokens = normed_vecs.keys()
	u = normed_vecs[x] - normed_vecs[y]

	normed_vecs_id = ray.put(normed_vecs)
	delta_id = ray.put(delta)
	# normed_vecs_id = normed_vecs
	# delta_id = delta
	neighbors = [compute_neighbors.remote(token, normed_vecs_id, delta=delta_id) for token in tokens]

	# assert tokens == tokens2
	print("Computing neighbors")
	neighbors = ray.get(neighbors)
	print("Finished computing neighbors")

	for i in range(len(neighbors)):
		a = neighbors[i][-1]
		ranking += [compute_score(u, a, b, normed_vecs_id, delta=delta_id) for b in neighbors[i][:-1]]
	print("Getting values multiprocessed")
	# ranking = ray.get(ranking)
	ranking = sorted(ranking, key=lambda pair: pair[2], reverse=True)
	return ranking


def save_ranking(ranking, fname):
	print(f"Saving ranking to {fname}")
	with open(fname, "w") as ff:
		yaml.dump(ranking, ff)


def load_ranking(fname):
	print(f"Loading ranking from {fname}")
	with open(fname, "r") as ff:
		ranking = yaml.load(ff)
	return ranking


def load_embeddings(fname):
	embeds = {}
	if fname.split('.')[-1] == 'vec':
		with open(fname, "r") as ff:
			for line in ff:
				line = line.split(" ")
				line[-1] = line[-1][:-1]
				token = line[0]
				vec = np.array([float(num) for num in line[1:]])
				if len(line) > 2:
					embeds[token] = np.array(vec)
		k = list(embeds.keys())[0]
		print(f"Finished loading embeds, total words: {len(embeds)}, size of embedding: {len(embeds[k])}")
	else:
		assert 0, f"error reading {fname}"
	return embeds


def show_ranking(ranking, top=20, filter_words=False):
	if filter_words:
		shown, count = set(), 0
		for pair in ranking:
			if pair[0] not in shown and pair[1] not in shown:
				print(pair)
				count += 1
				shown.add(pair[0])
				shown.add(pair[1])
			if count == top:
				break
	else:
		for pair in ranking[:top]:
			print(pair)


if __name__ == '__main__':
	args = parse_args()
	vocab = None
	if args.vocab:
		print(f"Loading vocab from {args.vocab}")
		with open(args.vocab, "r") as f:
			vocab = yaml.load(f)

	if args.show_ranking:
		pair_scores = load_ranking(args.show_ranking)
		show_ranking(pair_scores, top=args.top, filter_words=args.filter)
	else:
		if args.embeds_path:
			print(f"Loading embeddings from {args.embeds_path}")
			# with open(args.embeds_path, "r") as f:
			# 	embeds = yaml.load(f)
			embeds = load_embeddings(args.embeds_path)
			pair_scores = compute_score_analogy_pairs(args.x, args.y, embeds, vocab=vocab, delta=1)
			id_file = len(fnmatch.filter(os.listdir("./yaml_data/"), 'ranking_#*.yaml')) + 1
			save_ranking(pair_scores, f"./yaml_data/ranking_#{id_file}.yaml")
			show_ranking(pair_scores, top=args.top, filter_words=args.filter)
		else:
			print(f"Loading model {args.model}")
			model = api.load(models[args.model])
			pair_scores = compute_score_analogy_pairs(args.x, args.y, model.wv, vocab=vocab, delta=1)
			id_file = len(fnmatch.filter(os.listdir("./yaml_data/"), f'ranking_model_{args.model}_#*.yaml')) + 1
			save_ranking(pair_scores, f"./yaml_data/ranking_model_{args.model}_#{id_file}.yaml")
			show_ranking(pair_scores, top=args.top, filter_words=args.filter)

