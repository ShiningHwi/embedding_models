import faiss
import torch
import numpy as np
from sklearn.metrics import ndcg_score
from tqdm import tqdm
import os

def encode_texts(model, texts, batch_size):
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding texts", leave=False):
        batch = texts[i:i+batch_size]
        with torch.cuda.amp.autocast():
            emb = model.encode(batch, batch_size=batch_size, convert_to_tensor=False)
        emb = np.array(emb)
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)
        embeddings.append(emb)
    embeddings = np.vstack(embeddings)
    return embeddings.astype('float32')

def evaluate_model(model, queries, corpus, relevant_docs, top_k, batch_size, cache_path=None):
    if cache_path is not None and os.path.exists(cache_path):
        print(f"Loading cached corpus embeddings from {cache_path}")
        corpus_emb = np.load(cache_path)
    else:
        print("Encoding corpus texts...")
        corpus_emb = encode_texts(model, corpus, batch_size)
        if cache_path is not None:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            np.save(cache_path, corpus_emb)
    index = faiss.IndexFlatIP(corpus_emb.shape[1])
    index.add(corpus_emb)
    
    print("Encoding query texts...")
    query_emb = encode_texts(model, queries, batch_size)
    
    print("Corpus embedding shape:", corpus_emb.shape)
    print("Query embedding shape:", query_emb.shape)
    
    max_k = max(top_k)
    D, I = index.search(query_emb, max_k)
    
    results = {}
    for k in top_k:
        recalls, precisions, ndcgs, f1s = [], [], [], []
        for i, rel_docs in enumerate(tqdm(relevant_docs, desc=f"Evaluating top-{k} queries", leave=False)):
            pred_docs = I[i, :k].tolist()
            pred_scores = D[i, :k].tolist()
            
            recall_val = len(set(pred_docs) & set(rel_docs)) / len(rel_docs) if rel_docs else 0
            recalls.append(recall_val)
            
            precision_val = len(set(pred_docs) & set(rel_docs)) / k
            precisions.append(precision_val)
            
            if precision_val + recall_val > 0:
                f1_val = 2 * (precision_val * recall_val) / (precision_val + recall_val)
            else:
                f1_val = 0.0
            f1s.append(f1_val)
            
            y_true = np.array([[1 if doc in rel_docs else 0 for doc in pred_docs]])
            y_score = np.array([pred_scores])
            try:
                ndcg_val = ndcg_score(y_true, y_score, k=k)
            except ValueError:
                ndcg_val = 0.0
            ndcgs.append(ndcg_val)
            
        results[k] = {
            "recall": np.mean(recalls),
            "precision": np.mean(precisions),
            "ndcg": np.mean(ndcgs),
            "f1": np.mean(f1s)
        }
    return results
