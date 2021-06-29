import gensim


def run(data_path, board):
    model = gensim.models.KeyedVectors.load_word2vec_format(data_path, binary=True, limit=500000)
    return model.most_similar(positive=board["ally"], negative=board["opponent"], topn=10)
