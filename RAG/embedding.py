from sentence_transformers import SentenceTransformer

def load_embedding_model(model_name):
    return SentenceTransformer(model_name)
