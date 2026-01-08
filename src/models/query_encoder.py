from sentence_transformers import SentenceTransformer

class QueryEncoder:
    def __init__(self):
        self.model = SentenceTransformer("all-mpnet-base-v2")

    def encode(self, queries):
        return self.model.encode(queries, convert_to_tensor=True)