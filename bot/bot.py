import gensim
from sklearn.cluster import DBSCAN
import numpy


class Model:
    def __init__(self, data_path):
        # https://radimrehurek.com/gensim/models/word2vec.html
        print("Loading model...")
        self.model: gensim.models.KeyedVectors = gensim.models.KeyedVectors.load_word2vec_format(data_path, binary=True,
                                                                                                 limit=500000)

    # https://www.geeksforgeeks.org/difference-between-k-means-and-dbscan-clustering/
    # https://scikit-learn.org/stable/modules/clustering.html#dbscan
    # https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html#sklearn.cluster.DBSCAN.fit_predict
    # https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
    # a group is a group when there are `min_samples` other points `eps` distance away
    # label of -1 === noise
    # TODO: figure out good values for eps and min_samples that result in non- -1 values
    # eps=4 is way too high, <3 is too low, or maybe these words are just too sparse
    def cluster(self, words):
        vectors = [self.model.get_vector(w) for w in words]
        labelled = DBSCAN(eps=3.1, min_samples=1).fit_predict(vectors)

        clusters = {}
        for index, label in enumerate(labelled.tolist()):
            if str(label) not in clusters:
                clusters[str(label)] = []
            clusters[str(label)].append(words[index])
        return clusters

    def average_group_similarity(self, words):
        return numpy.mean([self.model.similarity(x, y) for x in words for y in words if x is not y])

    def rank_clusters(self, clusters):
        return {k: self.average_group_similarity(v) if len(v) > 1 else 0 for k, v in clusters.items()}

    def get_most_similar_cluster(self, clusters):
        ranked = self.rank_clusters(clusters)
        return clusters[max(ranked.keys(), key=lambda k: ranked[k])]

    def find_matches(self, positive, negative):
        return self.model.most_similar(positive, negative, topn=10)

    def run(self, board):
        print("Running bot...")

        clusters = self.cluster(board.get_all())
        most_similar_cluster = self.get_most_similar_cluster(clusters)
        sorted_words = board.sort(most_similar_cluster)
        matches = self.find_matches(sorted_words["positive"], sorted_words["negative"])
        # filter out words that are superstrings of words on the board or contain more than one word
        filtered = [m for m in matches if "_" not in m[0] and not board.is_superstring(m[0])][0]
        return filtered, sorted_words["positive"]
