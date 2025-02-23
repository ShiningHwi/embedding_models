import numpy as np
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
import torch

def encode_texts(model, texts, batch_size):
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding STS texts", leave=False):
        batch = texts[i:i+batch_size]
        with torch.cuda.amp.autocast():
            emb = model.encode(batch, batch_size=batch_size, convert_to_tensor=False)
        emb = np.array(emb)
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)
        embeddings.append(emb)
    embeddings = np.vstack(embeddings)
    return embeddings.astype('float32')

def evaluate_sts(model, sts_data, batch_size):
    n = len(sts_data)
    s1_list = [item[0] for item in sts_data]
    s2_list = [item[1] for item in sts_data]
    gt_scores = np.array([item[2] for item in sts_data])
    
    print("Encoding STS sentence pairs...")
    emb1 = encode_texts(model, s1_list, batch_size)
    emb2 = encode_texts(model, s2_list, batch_size)
    
    cosine_sim = []
    euclidean_sim = []
    manhattan_sim = []
    dot_sim = []
    
    for i in tqdm(range(n), desc="Evaluating STS pairs", leave=False):
        u = emb1[i]
        v = emb2[i]
        cos = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9)
        cosine_sim.append(cos)
        euc = -np.linalg.norm(u - v)
        euclidean_sim.append(euc)
        manh = -np.sum(np.abs(u - v))
        manhattan_sim.append(manh)
        dot = np.dot(u, v)
        dot_sim.append(dot)
    
    cosine_sim = np.array(cosine_sim)
    euclidean_sim = np.array(euclidean_sim)
    manhattan_sim = np.array(manhattan_sim)
    dot_sim = np.array(dot_sim)
    
    metrics = {}
    c_p, _ = pearsonr(cosine_sim, gt_scores)
    c_s, _ = spearmanr(cosine_sim, gt_scores)
    metrics["cosine_pearson"] = c_p
    metrics["cosine_spearman"] = c_s
    
    e_p, _ = pearsonr(euclidean_sim, gt_scores)
    e_s, _ = spearmanr(euclidean_sim, gt_scores)
    metrics["euclidean_pearson"] = e_p
    metrics["euclidean_spearman"] = e_s
    
    m_p, _ = pearsonr(manhattan_sim, gt_scores)
    m_s, _ = spearmanr(manhattan_sim, gt_scores)
    metrics["manhattan_pearson"] = m_p
    metrics["manhattan_spearman"] = m_s
    
    d_p, _ = pearsonr(dot_sim, gt_scores)
    d_s, _ = spearmanr(dot_sim, gt_scores)
    metrics["dot_pearson"] = d_p
    metrics["dot_spearman"] = d_s
    
    all_corrs = [c_p, c_s, e_p, e_s, m_p, m_s, d_p, d_s]
    metrics["AVG"] = np.mean(all_corrs)
    
    return metrics
