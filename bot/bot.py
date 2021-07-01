import gensim
from sklearn.cluster import DBSCAN, KMeans
import numpy
from math import ceil


class Model:
    def __init__(self, data_path):
        # https://radimrehurek.com/gensim/models/word2vec.html
        print("Loading model...")
        self.model: gensim.models.KeyedVectors = gensim.models.KeyedVectors.load_word2vec_format(data_path, binary=True,
                                                                                                 limit=500000)

    def words_to_vectors(self, words):
        return [self.model.get_vector(w) for w in words]

    def group_clusters(self, words, labelled_vectors):
        clusters = {}
        for index, label in enumerate(labelled_vectors.tolist()):
            if str(label) not in clusters:
                clusters[str(label)] = []
            clusters[str(label)].append(words[index])
        return clusters

    # https://www.geeksforgeeks.org/difference-between-k-means-and-dbscan-clustering/
    # https://scikit-learn.org/stable/modules/clustering.html#dbscan
    # https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html#sklearn.cluster.DBSCAN.fit_predict
    # https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
    # a group is a group when there are `min_samples` other points `eps` distance away
    # label of -1 === noise
    # TODO: figure out good values for eps and min_samples that result in non- -1 values
    # eps=4 is way too high, <3 is too low, or maybe these words are just too sparse
    def cluster_DBSCAN(self, words):
        print("Running DBSCAN...")
        vectors = self.words_to_vectors(words)
        labelled = DBSCAN(eps=3.1, min_samples=1).fit_predict(vectors)
        return self.group_clusters(words, labelled)

    def cluster_KMeans(self, words, cluster_size):
        print("Running KMeans...")
        vectors = self.words_to_vectors(words)
        # get # of clusters of the specified size
        # cluster_count = ceil(len(vectors) / min(len(vectors), cluster_size))
        cluster_count = len(vectors) - cluster_size + 1
        print("cluster count =", cluster_count)
        labelled = KMeans(n_clusters=cluster_count).fit_predict(vectors)
        return self.group_clusters(words, labelled)

    def average_group_similarity(self, words):
        return numpy.mean([self.model.similarity(x, y) for x in words for y in words if x is not y])

    def rank_clusters(self, clusters):
        return {k: self.average_group_similarity(v) if len(v) > 1 else 0 for k, v in clusters.items()}

    def get_most_similar_cluster(self, clusters):
        ranked = self.rank_clusters(clusters)
        return clusters[max(ranked.keys(), key=lambda k: ranked[k])]

    def find_matches(self, positive, negative):
        return self.model.most_similar(positive, negative, topn=10)

    def organize(self, board, clusters):
        most_similar_cluster = self.get_most_similar_cluster(clusters)
        sorted_words = board.sort(most_similar_cluster)
        matches = self.find_matches(sorted_words["positive"], board.get_other())
        # filter out words that are superstrings of words on the board or contain more than one word
        filtered = [m for m in matches if "_" not in m[0] and not board.is_superstring(m[0])][0]
        return filtered, sorted_words["positive"]

    def run_DBSCAN(self, board):
        clusters = self.cluster_DBSCAN(board.get_self())
        return self.organize(board, clusters)

    def run_KMeans(self, board, cluster_size):
        clusters = self.cluster_KMeans(board.get_self(), cluster_size)
        print(clusters)
        return self.organize(board, clusters)
