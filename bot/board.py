import json


def load_board(path):
    with open(path, "r") as board_file:
        data = board_file.read()
    return json.loads(data)


class Board:
    def __init__(self, path):
        data = load_board(path)
        self.positive = set(data["positive"])
        self.negative = set(data["negative"])
        self.neutral = set(data["neutral"])
        self.black = set(data["black"])

    def get_all(self):
        return list(self.positive | self.negative | self.neutral | self.black)

    def sort(self, words):
        sorted_words = {"positive": [], "negative": []}
        for w in words:
            if w in self.positive:
                sorted_words["positive"].append(w)
            else:
                sorted_words["negative"].append(w)
        return sorted_words

    def is_superstring(self, word):
        for w in self.get_all():
            if word.lower().strip() in w.lower():
                return True
        return False
