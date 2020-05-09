import numpy as np
import numpy.linalg as la
import itertools
import yaml
import argparse
import fnmatch
import os


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--show_ranking', type=str, default='', help="Path to YAML file with ranking results")
	parser.add_argument('--top', type=int, default=20, help="Number of results to show")
	parser.add_argument('--filter', type=bool, default=True, help="If filter True, only show pairs with different words. Default: True")
	parser.add_argument('--embeds_path', type=str, help="Path to YAML file with dictionary of embeddings")
	parser.add_argument('--x', type=int, help="Index of word")
	parser.add_argument('--y', type=int, help="Index of word")
	return parser.parse_args()


def compute_score_analogy_pairs(x, y, embeds, delta=1):
	"""
	Compute scores for analogy pairs (a, b) such that Norm(a-b) <= delta
	:param x: Index of word
	:param y: Index of word
	:param embeds: Dictionary of embeddings word_index -> embedding
	:param delta: threshold for semantic similarity of analogy pairs (delta = 1)
	:return: List of (a,b,score) tuples ranked by their score in non-increasing order, i.e., ranking[0] has higher score
	"""
	ranking = []
	normed_vecs = {}
	for idx, emb in embeds:
		normed_vecs[idx] = emb / la.norm(emb)
	idxs = normed_vecs.keys()

	u = normed_vecs[x] - normed_vecs[y]
	for (a, b) in itertools.product(idxs, idxs):
		v = normed_vecs[a] - normed_vecs[b]
		if la.norm(v) > delta:
			continue
		score = np.dot(u, v) / (np.norm(u) * np.norm(v))
		ranking.append((a, b, score))
	ranking = sorted(ranking, key=lambda pair: pair[2], reverse=True)
	return ranking


def save_ranking(ranking, fname):
	with open(fname, "w") as ff:
		yaml.dump(ranking, ff)


def load_ranking(fname):
	with open(fname, "r") as ff:
		ranking = yaml.load(ff)
	return ranking


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
	if args.show_ranking:
		pair_scores = load_ranking(args.show_ranking)
		show_ranking(pair_scores, top=args.top, filter_words=args.filter)
	else:
		with open(args.embeds_path, "r") as f:
			embeds = yaml.load(f)
		pair_scores = compute_score_analogy_pairs(args.x, args.y, embeds, delta=1)
		id_file = len(fnmatch.filter(os.listdir("./yaml_data/"), 'ranking_#*.yaml')) + 1
		save_ranking(pair_scores, f"./yaml_data/ranking_#{id_file}.yaml")
		show_ranking(pair_scores, top=args.top, filter_words=args.filter)
