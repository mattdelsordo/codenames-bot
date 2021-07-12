from pathlib import Path
from args import init_args
from board import Board
from bot import Model

BOARD_DIR = ""
DATA_DIR = "data"


def main():
    args = init_args()
    # data_path = Path(__file__).parent.parent.joinpath(DATA_DIR, args.data_path)
    board_path = Path(__file__).parent.parent.joinpath(BOARD_DIR, args.board_path)

    board = Board(board_path)
    model = Model(args.data_path)

    if args.dbscan:
        print(model.run_DBSCAN(board))
    elif args.kmeans:
        print(model.run_KMeans(board, args.kmeans))
    else:
        print("Unrecognized clustering algorithm")


if __name__ == '__main__':
    main()
