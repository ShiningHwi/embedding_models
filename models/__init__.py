import torch
from sentence_transformers import SentenceTransformer
from torch.cuda.amp import autocast

def load_model(model_name):
    model = SentenceTransformer(model_name)
    if torch.cuda.is_available():
        model = model.to('cuda')
    return model

def encode_sentences(model, sentences, batch_size=32):
    embeddings = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i+batch_size]
        with autocast():
            batch_emb = model.encode(batch, convert_to_tensor=True, device='cuda' if torch.cuda.is_available() else 'cpu')
        embeddings.append(batch_emb)
    return torch.cat(embeddings, dim=0)