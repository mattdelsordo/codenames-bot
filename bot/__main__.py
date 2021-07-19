from pathlib import Path
from args import init_args
from board import Board
from bot import DBSCANModel, KMeansModel, Model
from pynput import keyboard

BOARD_DIR = ""
DATA_DIR = "data"


def main():
    args = init_args()
    # data_path = Path(__file__).parent.parent.joinpath(DATA_DIR, args.data_path)
    board_path = Path(__file__).parent.parent.joinpath(BOARD_DIR, args.board_path)

    if args.dbscan:
        model = DBSCANModel(args.data_path)
    elif args.kmeans:
        model = KMeansModel(args.data_path)
    else:
        model = Model(args.data_path)

    def on_press(key):
        if key == keyboard.Key.enter:
            try:
                board = Board(board_path)
                print()
                model.run(board)
            except Exception as e:
                print("Execution error:", e)
        elif key == keyboard.Key.esc:
            return False

    with keyboard.Listener(on_press=on_press) as listener:
        print("Press 'enter' to generate a clue for the board. Press 'esc' to exit.")
        listener.join()


if __name__ == '__main__':
    main()
