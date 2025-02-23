MODELS = [
    "snunlp/KR-SBERT-V40K-klueNLI-augSTS",
    "jhgan/ko-sroberta-multitask",
    "upskyy/bge-m3-korean",
    "nlpai-lab/KURE-v1",
    "nlpai-lab/KoE5",
    "BM-K/KoSimCSE-roberta-multitask"
]

IR_DATASETS = [
    "Ko-StrategyQA",
    "markers_bm",
    "MLDR",
    "autorag-korean",
    "RAG-Evaluation-Dataset-KO"
]

STS_DATASETS = [
    "mteb/sts17-crosslingual-sts",
    "dkoterwa/kor-sts"
]

TOP_K = [1, 3, 5, 10]

DATASET_SPLITS = {
    "Ko-StrategyQA": "test",
    "markers_bm": "test",
    "MLDR": "test",
    "autorag-korean": "train", 
    "RAG-Evaluation-Dataset-KO": "test"
}
