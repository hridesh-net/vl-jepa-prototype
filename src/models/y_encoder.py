from sentence_transformers import SentenceTransformer

class YEncoder:
    def __init__(self, model="all-mpnet-base-v2"):
        self.model = SentenceTransformer(model)
    
    def encode(self, texts):
        return self.model.encode(texts, convert_to_tensor=True)