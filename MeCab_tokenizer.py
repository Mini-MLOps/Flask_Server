from konlpy.tag import MeCab


class MeCab_tokenizer:
    def __init__(self):
        self.mecab = MeCab()

    def tokenize(self, text):
        mecab_tokens = self.mecab.parse(text).split()
        # print("Mecab :", mecab_tokens)
        return mecab_tokens
