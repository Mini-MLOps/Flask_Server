from callback import callback
from gensim.models import Word2Vec
import numpy as np
import multiprocessing
import os


class Word2Vec_vectorizer:
    def model_train_save(
        self, text, vector_size, window, min_count, sg, epochs, model_path
    ):
        model = Word2Vec(
            text,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            sg=sg,
            epochs=epochs,
            workers=multiprocessing.cpu_count(),
            compute_loss=True,
            callbacks=[callback()],
        )
        if not os.path.exists(os.path.dirname(model_path)):
            os.makedirs(os.path.dirname(model_path))
        model.save(model_path)

    def model_load(self, model_path):
        return Word2Vec.load(model_path)

    def vectorize(self, model, text):
        text_vector_list = []
        for line in text:
            line_vector_list = []
            for word in line:
                if word in model.wv.index_to_key:
                    line_vector_list.append(model.wv[word])
            if line_vector_list:
                line_vector = np.mean(line_vector_list, axis=0)
                text_vector_list.append(line_vector.tolist())
        return text_vector_list
