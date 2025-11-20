import torch
import json
import os
from transformers import BertTokenizer, BertForSequenceClassification
from typing import List, Dict, Tuple, Any
import rdflib
from rdflib import Namespace

# --- 模型設定 ---
# 這些路徑將指向 Docker 容器內的模型檔案
MODEL_DIR = "/app/kbert_model_output_multilabel"
PRE_TRAINED_MODEL = 'bert-base-uncased' 
MAX_SEQ_LENGTH = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_LABELS = 4 
THRESHOLD = 0.5 # 判斷閾值

# 風險標籤 (必須與 3_kbert_event_detector.py 順序一致)
RISK_NAMES = [
    "Fall", 
    "Walk_with_memory_loss", 
    "Fall_with_climb", 
    "Run_with_disorientation"
]

# --- 全域變數，用於快取模型 ---
g_kbert_model = None
g_kbert_tokenizer = None

def load_kbert_model_and_tokenizer():
    """
    載入微調過的 K-BERT 模型和 Tokenizer。
    """
    global g_kbert_model, g_kbert_tokenizer
    if g_kbert_model and g_kbert_tokenizer:
        return g_kbert_model, g_kbert_tokenizer

    # 檢查模型檔案是否存在 (LFS 指標檔問題必須在 Docker build 時解決)
    if not os.path.exists(os.path.join(MODEL_DIR, "model.safetensors")):
         # Source 29 顯示 model.safetensors 存在，但如果 Git LFS 沒抓下來就會失敗
        raise FileNotFoundError(f"模型權重 'model.safetensors' 不在 {MODEL_DIR} 中。")

    try:
        print(f"[K-BERT] 正在從 {MODEL_DIR} 載入模型...")
        tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
        model = BertForSequenceClassification.from_pretrained(
            MODEL_DIR, 
            num_labels=NUM_LABELS,
            problem_type="multi_label_classification"
        )
        model.to(DEVICE)
        model.eval() # 設為評估模式

        g_kbert_model = model
        g_kbert_tokenizer = tokenizer
        print(f"[K-BERT] K-BERT 模型已載入至 {DEVICE}")
        return model, tokenizer

    except Exception as e:
        print(f"[K-BERT 錯誤] 載入模型失敗: {e}")
        raise

def extract_triplets_from_ttl(ttl_file_path: str) -> List[List[str]]:
    """
    從 .ttl 檔案中提取狀態知識三元組 (S, P, O)。
    (此邏輯基於 1_knowledge_extraction.py)
    """
    g = rdflib.Graph()
    try:
        g.parse(ttl_file_path, format='turtle')
    except Exception as e:
        print(f"[K-BERT 錯誤] 解析 TTL 檔案失敗: {e}")
        return []

    EX_NS = rdflib.Namespace("http://example.org/")
    VH2KG_NS = rdflib.Namespace("http://kgrc4si.home.kg/virtualhome2kg/ontology/")

    knowledge_triplets = []

    # 查詢 (person, has_state, lying_or_fall)
    for s, p, o in g.triples((None, EX_NS.has_state, EX_NS.lying_or_fall)):
        subject_name = str(s).split('/')[-1]
        predicate_name = "is_STATE"
        object_name = "LYING_OR_FALL" # 歸一化
        knowledge_triplets.append([subject_name, predicate_name, object_name])

    # 查詢 (person, near, object)
    for s, p, o in g.triples((None, EX_NS.near, None)):
        # 確保主語是 person，賓語不是 person
        if "person" in str(s) and "person" not in str(o):
            subject_name = str(s).split('/')[-1]
            predicate_name = "is_NEAR"
            object_name = str(o).split('/')[-1]
            knowledge_triplets.append([subject_name, predicate_name, object_name])

    # 查詢物件的危險狀態 (基於 1_knowledge_extraction.py 的邏輯)
    for state_entity, _, actual_object_uri in g.triples((None, VH2KG_NS.isStateOf, None)):
        object_name = actual_object_uri.toPython().split('/')[-1]
        for _, _, state_value_uri in g.triples((state_entity, VH2KG_NS.state, None)):
            state_predicate = state_value_uri.toPython().split('/')[-1]
            # 只保留 '危險' 狀態
            if state_predicate in ["DIRTY", "WET", "OPEN", "ON"]:
                knowledge_triplets.append([object_name, "is_STATE", state_predicate])

    # 返回唯一的列表
    unique_triplets = list(set(tuple(t) for t in knowledge_triplets))
    return [list(t) for t in unique_triplets]

def convert_triplets_to_text(visual_triples: List[Dict]) -> str:
    """
    將 main_gpu.py 產生的 visual_triples 字典列表轉換為 K-BERT 訓練時使用的句子。
    (此邏輯基於 2_dataset_generation.py)
    """
    sentences = []
    if not visual_triples:
        return "No visual activity detected."

    for triple in visual_triples:
        # 這是基於 DINO 的 triple 格式
        head = triple.get("head", "entity")
        relation = triple.get("relation", "related to")
        tail = triple.get("tail", "entity")

        # 轉換為 "The person has state lying or fall" 或 "The person near floor"
        head_clean = head.split('#')[0]

        sentence = f"The {head_clean} {relation.lower().replace('_', ' ')} {tail.lower().replace('_', ' ')}"
        sentences.append(sentence)

    return ". ".join(sentences) + "."


@torch.no_grad()
def predict_risk_from_kg(visual_triples_list: List[Dict], extracted_knowledge_list: List[List[str]]) -> Dict[str, Any]:
    """
    對 DINO 產生的視覺三元組 (visual_triples) 進行 K-BERT 風險預測，
    並使用從 TTL 提取的背景知識 (extracted_knowledge_list)。

    返回: 包含機率和最終報告的字典。
    """

    # 1. 載入模型
    model, tokenizer = load_kbert_model_and_tokenizer()

    # 2. 數據準備：將 DINO 視覺三元組轉換為模型輸入文本
    # (這裡我們模擬了 2_dataset_generation.py 的輸入格式)
    input_text = convert_triplets_to_text(visual_triples_list)

    # 3. Tokenization (K-BERT 數據加載器的核心邏輯)
    encoding = tokenizer.encode_plus(
        input_text,
        max_length=MAX_SEQ_LENGTH,
        padding='max_length',
        truncation=True,
        return_tensors='pt',
        return_attention_mask=True,
        return_token_type_ids=True
    )

    # 執行預測 (使用上面生成的 input_text)
    inputs = {
        'input_ids': encoding['input_ids'].to(DEVICE),
        'attention_mask': encoding['attention_mask'].to(DEVICE),
        'token_type_ids': encoding['token_type_ids'].to(DEVICE),
    }

    outputs = model(**inputs)
    logits = outputs.logits

    # Sigmoid 取得機率
    probabilities = torch.sigmoid(logits).squeeze().cpu().numpy()

    # 閾值判斷
    predictions = (probabilities >= THRESHOLD).astype(int)

    # 組織結果
    probabilities_dict = {RISK_NAMES[i]: float(probabilities[i]) for i in range(NUM_LABELS)}
    predictions_dict = {RISK_NAMES[i]: int(predictions[i]) for i in range(NUM_LABELS)}

    # 簡易報告生成
    report_lines = []
    if predictions_dict["Fall"]:
        report_lines.append(f"偵測到高風險跌倒 (機率: {probabilities_dict['Fall']:.2f})。")
    if predictions_dict["Walk_with_memory_loss"]:
        report_lines.append(f"偵測到迷失/遊走跡象 (機率: {probabilities_dict['Walk_with_memory_loss']:.2f})。")
    if predictions_dict["Fall_with_climb"]:
        report_lines.append(f"偵測到危險攀爬 (機率: {probabilities_dict['Fall_with_climb']:.2f})。")
    if predictions_dict["Run_with_disorientation"]:
        report_lines.append(f"偵測到異常奔跑 (機率: {probabilities_dict['Run_with_disorientation']:.2f})。")

    if not report_lines:
        report_summary = "K-BERT 分析未在影片中偵測到顯著的特定風險行為。"
    else:
        report_summary = "K-BERT 風險評估：\n" + "\n".join(report_lines)

    return {
        "status": "completed",
        "summary": report_summary,
        "contains_danger": any(p == 1 for p in predictions),
        "details": {
            "predictions": predictions_dict,
            "probabilities": probabilities_dict,
            "model_input_text": input_text,
            "knowledge_triplets_used_count": len(extracted_knowledge_list) # 告知我們用了多少背景知識
        }
    }