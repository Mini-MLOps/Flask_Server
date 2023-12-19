# -*- coding: utf-8 -*-

import os
import openai


class GPT_embedding:
    def __init__(self):
        api_version = "2023-03-15-preview"

        os.environ["AZURE_OPENAI_ENDPOINT"] = "https://student-oai.openai.azure.com/"
        os.environ["AZURE_OPENAI_API_KEY"] = "8ff210f0e95f4a00aa652b331f1b9c28"

        openai.api_type = "azure"
        openai.api_version = api_version

    def vectorize(self, text):
        return openai.embeddings.create(model="text-embedding-ada-002", input=text).data[0].embedding
