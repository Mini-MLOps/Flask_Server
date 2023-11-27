import sentencepiece as spm


class Sentencepiece_tokenizer:
    def model_train(self, input, model_prefix, vocab_size):
        spm.SentencePieceTrainer.Train(
            f"--input={input} --model_prefix={model_prefix} --vocab_size={vocab_size}"
        )

    def model_load(self, model_path):
        model = spm.SentencePieceProcessor()
        model.load(model_path)
        return model

    def tokenize(self, model, text):
        tokens = model.encode_as_pieces(text)
        tokens_without_space = [token.replace("▁", "") for token in tokens]
        tokens_without_empty = [token for token in tokens_without_space if token != ""]
        # print("Sentencepiece :", tokens_without_empty)
        return tokens_without_empty
