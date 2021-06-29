import argparse


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", "-d", dest="data_path", type=str, required=True)
    parser.add_argument("--board", "-b", dest="board_path", type=str, required=True)
    return parser.parse_args()
