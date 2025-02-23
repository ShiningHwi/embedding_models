import os
import json
import pandas as pd
import streamlit as st

st.set_page_config(layout="wide")

def load_ir_results(results_dir):
    ir_data = []
    for model in os.listdir(results_dir):
        model_dir = os.path.join(results_dir, model)
        if not os.path.isdir(model_dir):
            continue
        for file in os.listdir(model_dir):
            if file.endswith("_ir.json"):
                dataset_name = file.replace("_ir.json", "")
                file_path = os.path.join(model_dir, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    res = json.load(f)
                for topk_key in res:
                    metrics = res[topk_key]
                    ir_data.append({
                        "Model": model,
                        "Dataset": dataset_name,
                        "TopK": topk_key,
                        "Recall": metrics.get("recall"),
                        "Precision": metrics.get("precision"),
                        "NDCG": metrics.get("ndcg"),
                        "F1": metrics.get("f1")
                    })
    return pd.DataFrame(ir_data)

def load_sts_results(results_dir):
    sts_data = []
    for model in os.listdir(results_dir):
        model_dir = os.path.join(results_dir, model)
        if not os.path.isdir(model_dir):
            continue
        for file in os.listdir(model_dir):
            if file.endswith("_sts.json"):
                dataset_name = file.replace("_sts.json", "")
                file_path = os.path.join(model_dir, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    res = json.load(f)
                sts_data.append({
                    "Model": model,
                    "Dataset": dataset_name,
                    "AVG": res.get("AVG"),
                    "Cosine Pearson": res.get("cosine_pearson"),
                    "Cosine Spearman": res.get("cosine_spearman"),
                    "Euclidean Pearson": res.get("euclidean_pearson"),
                    "Euclidean Spearman": res.get("euclidean_spearman"),
                    "Manhattan Pearson": res.get("manhattan_pearson"),
                    "Manhattan Spearman": res.get("manhattan_spearman"),
                    "Dot Pearson": res.get("dot_pearson"),
                    "Dot Spearman": res.get("dot_spearman")
                })
    return pd.DataFrame(sts_data)

def save_and_display_df(df, title, csv_filename):
    st.markdown(f"### {title}")
    st.dataframe(df, use_container_width=True)
    df.to_csv(csv_filename, index=False)
    st.write(f"Saved CSV: {csv_filename}")

def app():
    results_dir = "results" 
    st.title("벤치마크 결과 비교 리더보드")
    
    st.markdown("## IR Results")
    ir_df = load_ir_results(results_dir)
    if not ir_df.empty:
        ir_pivot = ir_df.pivot_table(index="Model", columns=["Dataset", "TopK"],
                                      values=["Recall", "Precision", "NDCG", "F1"])
        st.dataframe(ir_pivot, use_container_width=True)
        save_and_display_df(ir_df, "IR Raw Results", "ir_results.csv")
    else:
        st.write("IR 결과 데이터가 없습니다.")
    
    st.markdown("## STS Results")
    sts_df = load_sts_results(results_dir)
    if not sts_df.empty:
        sts_pivot = sts_df.pivot_table(index="Model", columns="Dataset")
        st.dataframe(sts_pivot, use_container_width=True)
        save_and_display_df(sts_df, "STS Raw Results", "sts_results.csv")
    else:
        st.write("STS 결과 데이터가 없습니다.")

if __name__ == "__main__":
    app()
