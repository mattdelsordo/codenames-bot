import json


def load_board(path):
    with open(path, "r") as board_file:
        data = board_file.read()
    return json.loads(data)


class Board:
    def __init__(self, path):
        data = load_board(path)
        self.turn = data["turn"]
        self.red = set(data["red"])
        self.blue = set(data["blue"])
        self.white = set(data["white"])
        self.black = set(data["black"])
        self.guess = int(data["guess"])

    def get_own(self):
        return self.blue if self.turn == "blue" else self.red

    def get_opponent(self):
        return self.red if self.turn == "blue" else self.blue

    def get_good_options(self):
        return list(self.get_own())

    def get_bad_options(self):
        return list(self.get_opponent() | self.white | self.black)

    def get_all(self):
        return self.get_good_options() + self.get_bad_options()

    # def sort(self, words):
    #     sorted_words = {"positive": [], "negative": []}
    #     for w in words:
    #         if w in self.positive:
    #             sorted_words["positive"].append(w)
    #         else:
    #             sorted_words["negative"].append(w)
    #     return sorted_words

    def is_superstring(self, word):
        for w in self.get_all():
            stripped_word = word.lower().strip()
            stripped_w = w.lower()
            if stripped_word in stripped_w or stripped_w in stripped_word:
                return True
        return False
