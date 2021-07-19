import gensim
import gensim.downloader as api
from sklearn.cluster import DBSCAN, KMeans
import numpy
import time


class Model:
    def __init__(self, data_path):
        start = time.time()
        # https://radimrehurek.com/gensim/models/word2vec.html
        print("Loading model", data_path)
        # https://github.com/RaRe-Technologies/gensim-data
        self.model: gensim.models.KeyedVectors = api.load(data_path)
        end = time.time()
        print("Model loaded in", end - start, "seconds")

    def words_to_vectors(self, words):
        return [self.model.get_vector(w) for w in words]

    def group_clusters(self, words, labelled_vectors):
        clusters = {}
        for index, label in enumerate(labelled_vectors.tolist()):
            if str(label) not in clusters:
                clusters[str(label)] = []
            clusters[str(label)].append(words[index])
        return list(clusters.values())

    def cluster(self, words, guess):
        print("Default clustering for", words)
        cluster = words.copy()
        while len(cluster) > min(guess, len(words)):
            no_match = self.model.doesnt_match(cluster)
            cluster.remove(no_match)
        # print("cluster", cluster)
        # return numpy.array([(c, 0) for c in cluster])
        return cluster

    def average_group_similarity(self, words):
        return numpy.mean([self.model.similarity(x, y) for x in words for y in words if x is not y])

    def rank_clusters(self, clusters):
        return [self.average_group_similarity(v) if len(v) > 1 else 0 for v in clusters]

    def get_most_similar_cluster(self, clusters):
        ranked = self.rank_clusters(clusters)
        return clusters[ranked.index(max(ranked))]

    # def find_matches(self, positive, negative):
    #     # https://radimrehurek.com/gensim/models/keyedvectors.html
    #     # TODO: adding negatives seems to bias model against ENGLISH, find english only model
    #     return self.model.most_similar(negative=negative, topn=10)
    #     # return self.model.most_similar(positive, topn=10)

    def filter_words(self, matches, board):
        # remove words that are superstrings of words on the board or have spaces
        filtered = [m for m in matches if "_" not in m and not board.is_superstring(m)]
        # TODO: remove function words?
        # might be useful: https://github.com/dariusk/corpora/tree/master/data/words
        return filtered

    def find_farthest_word(self, matches, negative_words):
        distances = [(m, self.model.distances(m, negative_words)) for m in matches]
        farthest = max(distances, key=lambda k: numpy.mean(k[1]))
        return farthest[0]

    def find_most_similar(self, words):
        similar = self.model.most_similar(positive=words, topn=10)
        return [s[0] for s in similar]

    def run(self, board):
        words = board.get_good_options()
        # 1. find best cluster of options
        best_cluster = self.cluster(words, board.guess)
        # grouped = self.group_clusters(words, clusters)
        # print("Clusters of size > 1:")
        # for c in [cluster for cluster in grouped if len(cluster) > 1]:
        #     print(c)
        # best_cluster = self.get_most_similar_cluster(grouped)
        print("For cluster", best_cluster)

        # 2. find matches for cluster
        # sorted_words = board.sort(best_cluster)
        matches = self.find_most_similar(best_cluster)

        # 3. filter/sanitize matches
        filtered = self.filter_words(matches, board)
        print("Filtered matches:", filtered)

        # 4. return match thats farthest from the negative words
        farthest = self.find_farthest_word(filtered, board.get_bad_options())
        print("GUESS:", farthest)


class KMeansModel(Model):
    def __init__(self, data_path, cluster_size):
        super(KMeansModel, self).__init__(data_path)
        self.cluster_size = cluster_size

    def cluster(self, words, guess):
        print("Running KMeans...")
        vectors = self.words_to_vectors(words)
        # get # of groups, should leave enough for one group of N
        cluster_count = len(vectors) - guess + 1
        labelled = KMeans(n_clusters=cluster_count).fit_predict(vectors)
        return labelled


class DBSCANModel(Model):
    # https://www.geeksforgeeks.org/difference-between-k-means-and-dbscan-clustering/
    # https://scikit-learn.org/stable/modules/clustering.html#dbscan
    # https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html#sklearn.cluster.DBSCAN.fit_predict
    # https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
    # a group is a group when there are `min_samples` other points `eps` distance away
    # label of -1 === noise
    # TODO: figure out good values for eps and min_samples that result in non- -1 values
    # eps=4 is way too high, <3 is too low, or maybe these words are just too sparse
    def cluster(self, words, guess):
        print("Running DBSCAN...")
        vectors = self.words_to_vectors(words)
        labelled = DBSCAN(eps=3.1, min_samples=1).fit_predict(vectors)
        return labelled
