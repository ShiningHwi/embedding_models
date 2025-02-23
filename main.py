import os
import json
from config import MODELS, IR_DATASETS, STS_DATASETS, TOP_K
from models import load_model
from data_loaders import load_data, load_sts_data
from eval import evaluate_model
from eval_sts import evaluate_sts
from tqdm import tqdm

def main():
    fixed_batch_size = 1
    for model_name in tqdm(MODELS, desc="Evaluating models"):
        print(f"\nEvaluating model: {model_name}")
        model = load_model(model_name)
        safe_model_name = model_name.replace("/", "_")
        os.makedirs(f"results/{safe_model_name}", exist_ok=True)
        print(f"사용 배치 사이즈: {fixed_batch_size}")
        for dataset_name in tqdm(IR_DATASETS, desc="IR datasets", leave=False):
            print(f"  IR Dataset: {dataset_name}")
            queries, corpus, relevant_docs = load_data(dataset_name)
            cache_path = f"cache/{safe_model_name}_{dataset_name}_corpus_emb.npy"
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            results = evaluate_model(model, queries, corpus, relevant_docs, TOP_K, batch_size=fixed_batch_size, cache_path=cache_path)
            result_path = f"results/{safe_model_name}/{dataset_name}_ir.json"
            os.makedirs(os.path.dirname(result_path), exist_ok=True)
            with open(result_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=4, ensure_ascii=False)
            print(f"    Results saved to {result_path}")
        for dataset_name in tqdm(STS_DATASETS, desc="STS datasets", leave=False):
            print(f"  STS Dataset: {dataset_name}")
            sts_data = load_sts_data(dataset_name)
            sts_results = evaluate_sts(model, sts_data, batch_size=fixed_batch_size)
            safe_dataset_name = dataset_name.replace("/", "_")
            result_path = f"results/{safe_model_name}/{safe_dataset_name}_sts.json"
            os.makedirs(os.path.dirname(result_path), exist_ok=True)
            with open(result_path, "w", encoding="utf-8") as f:
                json.dump(sts_results, f, indent=4, ensure_ascii=False)
            print(f"    Results saved to {result_path}")

if __name__ == "__main__":
    main()
