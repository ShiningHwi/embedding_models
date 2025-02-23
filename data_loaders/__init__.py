from datasets import load_dataset
import os

def debug_dataset_info(ds, ds_name, config_name, split):
    print(f"[DEBUG] {ds_name} - config: {config_name}, split: {split}")
    if isinstance(ds, dict):
        if split in ds:
            data = ds[split]
            print(f"  Keys: {list(data[0].keys()) if data and isinstance(data, list) else 'N/A'}")
            print(f"  # of examples: {len(data)}")
        else:
            print(f"  Split '{split}' not found. Available splits: {list(ds.keys())}")
    else:
        try:
            print(f"  Dataset features: {ds.features}")
            print(f"  # of examples: {len(ds)}")
        except Exception as e:
            print(f"  Could not extract detailed info: {e}")

def load_data(dataset_name):
    if dataset_name == "Ko-StrategyQA":
        queries_dataset = load_dataset("taeminlee/Ko-StrategyQA", "queries")
        debug_dataset_info(queries_dataset, "Ko-StrategyQA", "queries", "queries")
        queries = [q["text"] for q in queries_dataset["queries"]]
        
        corpus_dataset = load_dataset("taeminlee/Ko-StrategyQA", "corpus")
        debug_dataset_info(corpus_dataset, "Ko-StrategyQA", "corpus", "corpus")
        corpus = [c["text"] for c in corpus_dataset["corpus"]]
        corpus_ids = [c["_id"] for c in corpus_dataset["corpus"]]
        corpus_id_to_index = {cid: i for i, cid in enumerate(corpus_ids)}
        
        default_dataset = load_dataset("taeminlee/Ko-StrategyQA", "default")
        debug_dataset_info(default_dataset, "Ko-StrategyQA", "default", "dev")
        query_to_corpus = {}
        for row in default_dataset["dev"]:
            query_id = row["query-id"]
            corpus_id = row["corpus-id"]
            if query_id not in query_to_corpus:
                query_to_corpus[query_id] = []
            if corpus_id in corpus_id_to_index:
                query_to_corpus[query_id].append(corpus_id_to_index[corpus_id])
            else:
                print(f"[DEBUG] Ko-StrategyQA: corpus_id {corpus_id} not found in corpus_id_to_index")
        
        query_ids = [q["_id"] for q in queries_dataset["queries"]]
        relevant_docs = [query_to_corpus.get(q_id, []) for q_id in query_ids]
        
        return queries, corpus, relevant_docs

    elif dataset_name == "markers_bm":
        queries_dataset = load_dataset("yjoonjang/markers_bm", "queries")
        debug_dataset_info(queries_dataset, "markers_bm", "queries", "queries")
        queries = [q["text"] for q in queries_dataset["queries"]]
        
        corpus_dataset = load_dataset("yjoonjang/markers_bm", "corpus")
        debug_dataset_info(corpus_dataset, "markers_bm", "corpus", "corpus")
        corpus = [c["text"] for c in corpus_dataset["corpus"]]
        corpus_ids = [c["_id"] for c in corpus_dataset["corpus"]]
        corpus_id_to_index = {cid: i for i, cid in enumerate(corpus_ids)}
        
        default_dataset = load_dataset("yjoonjang/markers_bm", "default")
        debug_dataset_info(default_dataset, "markers_bm", "default", "test")
        query_to_corpus = {}
        for row in default_dataset["test"]:
            query_id = row["query-id"]
            corpus_id = row["corpus-id"]
            if query_id not in query_to_corpus:
                query_to_corpus[query_id] = []
            if corpus_id in corpus_id_to_index:
                query_to_corpus[query_id].append(corpus_id_to_index[corpus_id])
            else:
                print(f"[DEBUG] markers_bm: corpus_id {corpus_id} not found in corpus_id_to_index")
        
        query_ids = [q["_id"] for q in queries_dataset["queries"]]
        relevant_docs = [query_to_corpus.get(q_id, []) for q_id in query_ids]
        
        return queries, corpus, relevant_docs

    elif dataset_name == "MLDR":
        dataset = load_dataset("Shitao/MLDR", "ko", split="test")
        debug_dataset_info(dataset, "MLDR", "ko", "test")
        queries = [item["query"] for item in dataset]
        corpus_ds = load_dataset("Shitao/MLDR", "corpus-ko", split="corpus")
        debug_dataset_info(corpus_ds, "MLDR", "corpus-ko", "corpus")
        corpus = [item["text"] for item in corpus_ds]
        docid_to_index = {item["docid"]: i for i, item in enumerate(corpus_ds)}
        relevant_docs = [
            [docid_to_index[p["docid"]] for p in item["positive_passages"] if p["docid"] in docid_to_index]
            for item in dataset
        ]
        return queries, corpus, relevant_docs

    elif dataset_name == "autorag-korean":
        dataset = load_dataset("sionic-ai/autorag-korean", "AutoRAG")
        if "queries" in dataset and "corpus" in dataset:
            debug_dataset_info(dataset, "autorag-korean", "AutoRAG", "queries")
            debug_dataset_info(dataset, "autorag-korean", "AutoRAG", "corpus")
        else:
            print("[DEBUG] autorag-korean: 예상하는 'queries' 또는 'corpus' split이 없습니다.")
        queries = [q["text"] for q in dataset["queries"]]
        corpus = [c["text"] for c in dataset["corpus"]]
        relevant_docs = [[0] for _ in queries]
        return queries, corpus, relevant_docs

    elif dataset_name == "RAG-Evaluation-Dataset-KO":
        dataset = load_dataset("allganize/RAG-Evaluation-Dataset-KO", split="test")
        debug_dataset_info(dataset, "RAG-Evaluation-Dataset-KO", "default", "test")
        queries = [item["question"] for item in dataset]
        corpus = [item["target_answer"] for item in dataset]
        relevant_docs = [[i] for i in range(len(queries))]
        return queries, corpus, relevant_docs

    else:
        raise ValueError(f"지원되지 않는 데이터셋: {dataset_name}")

def load_sts_data(dataset_name):
    if dataset_name == "mteb/sts17-crosslingual-sts":
        ds = load_dataset("mteb/sts17-crosslingual-sts", "ko-ko", split="test")
        data = [(item["sentence1"], item["sentence2"], item["score"]) for item in ds]
        return data
    elif dataset_name == "dkoterwa/kor-sts":
        ds = load_dataset("dkoterwa/kor-sts", split="test")
        data = [(item["sentence1"], item["sentence2"], item["score"]) for item in ds]
        return data
    else:
        raise ValueError(f"지원되지 않는 STS 데이터셋: {dataset_name}")

if __name__ == "__main__":
    for ds_name in ["Ko-StrategyQA", "markers_bm", "MLDR", "autorag-korean", "RAG-Evaluation-Dataset-KO"]:
        print(f"\n[DEBUG] Loading IR dataset: {ds_name}")
        try:
            queries, corpus, relevant_docs = load_data(ds_name)
            print(f"Loaded {ds_name}: {len(queries)} queries, {len(corpus)} corpus documents, {len(relevant_docs)} relevant_docs entries")
        except Exception as e:
            print(f"[DEBUG] Error loading {ds_name}: {e}")
    
    for ds_name in ["mteb/sts17-crosslingual-sts", "dkoterwa/kor-sts"]:
        print(f"\n[DEBUG] Loading STS dataset: {ds_name}")
        try:
            sts_data = load_sts_data(ds_name)
            print(f"Loaded {ds_name}: {len(sts_data)} sentence pairs")
        except Exception as e:
            print(f"[DEBUG] Error loading {ds_name}: {e}")
