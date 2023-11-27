import base64
import numpy as np


class Encode_Decode:
    def encode(self, vector_list):
        return base64.b85encode(np.float32(vector_list).tobytes()).decode()

    def decode(self, vector_string):
        return list(np.frombuffer(base64.b85decode(vector_string), dtype=np.float32))
