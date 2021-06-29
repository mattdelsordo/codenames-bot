import json
from pathlib import Path
from args import init_args
from bot import run

BOARD_DIR = ""
DATA_DIR = "data"


def load_board(path):
    with open(path, "r") as board_file:
        data = board_file.read()
    return json.loads(data)


def main():
    args = init_args()
    data_path = Path(__file__).parent.parent.joinpath(DATA_DIR, args.data_path)
    board_path = Path(__file__).parent.parent.joinpath(BOARD_DIR, args.board_path)

    result = run(data_path, load_board(board_path))
    print(result)


if __name__ == '__main__':
    main()
