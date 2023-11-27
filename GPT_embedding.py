# -*- coding: utf-8 -*-

import openai


class GPT_embedding:
    def __init__(self, api_key, model):
        openai.api_key = api_key
        self.model = model

    def vectorize(self, text):
        return openai.Embedding.create(model=self.model, input=text).data[0].embedding
