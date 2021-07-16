from pathlib import Path
from args import init_args
from board import Board
from bot import DBSCANModel, KMeansModel

BOARD_DIR = ""
DATA_DIR = "data"


def main():
    args = init_args()
    # data_path = Path(__file__).parent.parent.joinpath(DATA_DIR, args.data_path)
    board_path = Path(__file__).parent.parent.joinpath(BOARD_DIR, args.board_path)

    board = Board(board_path)

    if args.dbscan:
        model = DBSCANModel(args.data_path)
        print(model.run(board))
    elif args.kmeans:
        model = KMeansModel(args.data_path, args.kmeans)
        print(model.run(board))
    else:
        print("Unrecognized clustering algorithm")


if __name__ == '__main__':
    main()
