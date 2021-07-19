import argparse


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", "-d", dest="data_path", type=str, required=True)
    parser.add_argument("--board", "-b", dest="board_path", type=str, required=True)
    parser.add_argument("--kmeans", dest="kmeans", type=bool)
    parser.add_argument("--dbscan", dest="dbscan", type=bool)
    return parser.parse_args()
