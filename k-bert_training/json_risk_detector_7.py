# 功能: 讀取 6_ttl_to_json_converter.py 產生的 JSON 動作序列，並使用 K-BERT 進行多標籤風險偵測。

import os
import json
import torch
import torch.nn as nn
from typing import List, Dict, Tuple
import numpy as np
import argparse
from transformers import BertTokenizer, BertForSequenceClassification

# ==================================
# K-BERT 模型配置 (請依據您的訓練結果調整)
# ==================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# 確保這個路徑指向您打包的 K-BERT 模型資料夾
KBERT_MODEL_OUTPUT_DIR = "./kbert_model_output_multilabel" 
MAX_SEQ_LENGTH = 128
THRESHOLD = 0.5 
# 請將這裡的風險名稱替換為您 K-BERT 模型訓練時使用的實際標籤順序
RISK_NAMES = [
    "Fall", 
    "Walk_with_memory_loss", 
    "Fall_with_climb", 
    "Run_with_disorientation"
]

# 全域物件（Lazy-init）
kbert_tokenizer = None
kbert_model = None

# ==================================
# 輔助函數
# ==================================

def initialize_kbert_model():
    """載入 K-BERT 模型和分詞器。"""
    global kbert_tokenizer, kbert_model
    if kbert_model is None:
        print(f"[INFO] 開始載入 K-BERT 模型 (裝置: {DEVICE})...")
        try:
            if not os.path.exists(KBERT_MODEL_OUTPUT_DIR):
                 raise FileNotFoundError(f"找不到 K-BERT 模型於 {KBERT_MODEL_OUTPUT_DIR}。請檢查路徑。")
            
            kbert_tokenizer = BertTokenizer.from_pretrained(KBERT_MODEL_OUTPUT_DIR)
            kbert_model = BertForSequenceClassification.from_pretrained(KBERT_MODEL_OUTPUT_DIR)
            kbert_model.to(DEVICE)
            kbert_model.eval()
            print("[INFO] K-BERT 模型載入完成。")
        except Exception as e:
            print(f"[ERROR] K-BERT 模型載入失敗: {e}")
            kbert_model = None
            kbert_tokenizer = None
            raise

def actions_to_text(actions: List[str]) -> str:
    """將動作列表轉換為 K-BERT 可接受的文本序列。"""
    sentences = []
    for action in actions:
        # 移除動作名稱後的數字 (例如: Get_out_of_bed1 -> Get_out_of_bed)
        action_name = ''.join([i for i in action if not i.isdigit()])
        # 將底線替換為空格，並小寫化
        processed_action = action_name.lower().replace('_', ' ')
        
        if processed_action:
             sentences.append(f"The person {processed_action}")
        else:
             sentences.append(f"The person performs action {action.lower().replace('_', ' ')}")
             
    # 返回一個以句點分隔的單一長字串
    return ". ".join(sentences) + "."

@torch.no_grad()
def detect_risk_from_sequence(action_sequence: List[str]) -> Tuple[List[int], List[float]]:
    """
    使用 K-BERT 預測給定動作序列的風險類別。
    返回: (預測向量, 機率向量)
    """
    global kbert_model, kbert_tokenizer
    if kbert_model is None:
        print("[WARNING] K-BERT 模型未載入，返回預設安全結果。")
        return [0] * len(RISK_NAMES), [0.0] * len(RISK_NAMES)

    input_text = actions_to_text(action_sequence)
    
    print(f"[DEBUG] K-BERT 輸入文本: {input_text[:100]}...")

    # 將文本編碼為模型輸入
    encoding = kbert_tokenizer.encode_plus(
        input_text,
        max_length=MAX_SEQ_LENGTH,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    # 進行預測
    try:
        inputs = {
            'input_ids': encoding['input_ids'].to(DEVICE),
            'attention_mask': encoding['attention_mask'].to(DEVICE),
            'token_type_ids': encoding['token_type_ids'].to(DEVICE),
        }
        outputs = kbert_model(**inputs)
        
        # 使用 Sigmoid 函式取得機率 (多標籤分類)
        probabilities = torch.sigmoid(outputs.logits).squeeze().cpu().numpy()
        
        # 根據閾值決定最終的預測類別
        prediction_vector = (probabilities >= THRESHOLD).astype(int)
        
        return prediction_vector.tolist(), probabilities.tolist()

    except Exception as e:
        print(f"[ERROR] K-BERT 預測失敗: {e}")
        return [0] * len(RISK_NAMES), [0.0] * len(RISK_NAMES)

# ==================================
# 核心處理函數
# ==================================

def analyze_json_for_risk(json_path: str):
    """
    讀取 JSON 檔案，提取動作序列，並進行 K-BERT 風險分析。
    """
    if not os.path.exists(json_path):
        print(f"[FATAL] 找不到輸入 JSON 檔案: {json_path}")
        return None
        
    print(f"[INFO] 讀取 JSON 檔案: {os.path.basename(json_path)}")
    
    # 1. 讀取並解析 JSON
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # 提取核心動作序列
        action_sequence = data.get("data", {}).get("activities", [])
        
        if not action_sequence:
            print("[WARNING] JSON 檔案中未找到 'data.activities' 列表或列表為空。")
            return {
                "status": "completed", 
                "summary": "無動作序列輸入，視為安全。", 
                "contains_danger": False
            }
        
    except Exception as e:
        print(f"[ERROR] 解析 JSON 檔案失敗: {e}")
        return {
            "status": "error", 
            "summary": f"JSON 解析錯誤: {str(e)}", 
            "contains_danger": False
        }

    print(f"[INFO] 提取動作序列 ({len(action_sequence)} 條): {action_sequence}")
    
    # 2. 進行 K-BERT 預測
    prediction_vector, probabilities_vector = detect_risk_from_sequence(action_sequence)
    
    # 3. 整理輸出結果
    
    # 確保輸出長度一致，防止索引錯誤
    min_len = min(len(RISK_NAMES), len(prediction_vector), len(probabilities_vector))
    
    specific_risks = {
        RISK_NAMES[i]: {"detected": int(prediction_vector[i]), "probability": float(probabilities_vector[i])} 
        for i in range(min_len)
    }
    
    risky_count = sum(prediction_vector)
    contains_danger = risky_count > 0
    risk_summary = ", ".join([name for name, val in specific_risks.items() if val['detected'] == 1])
    
    summary = f"偵測到 {risky_count} 種風險：{risk_summary}" if contains_danger else "未偵測到風險。"
    
    result = {
        "status": "completed",
        "summary": summary,
        "contains_danger": contains_danger,
        "input_json_file": os.path.basename(json_path),
        "input_activities": action_sequence,
        "risk_details": specific_risks
    }
    
    print("-" * 30)
    print(f"[RESULT] {summary}")
    for name, detail in specific_risks.items():
        print(f"  - {name}: 偵測 {'是' if detail['detected'] == 1 else '否'} (機率: {detail['probability']:.4f})")
    print("-" * 30)
    
    return result

# ==================================
# 腳本執行入口
# ==================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="K-BERT risk detection using JSON action sequence input.")
    parser.add_argument(
        '--json_file', 
        type=str, 
        required=True, 
        help='Path to the JSON file containing the activities list (from 6_ttl_to_json_converter.py).'
    )
    args = parser.parse_args()

    try:
        initialize_kbert_model()
        analysis_result = analyze_json_for_risk(args.json_file)
        
        # 可選：將分析結果輸出為另一個 JSON 檔案
        if analysis_result:
            output_filename = os.path.splitext(os.path.basename(args.json_file))[0] + "_risk_report.json"
            output_path = os.path.join(os.path.dirname(args.json_file), output_filename)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(analysis_result, f, indent=4, ensure_ascii=False)
            print(f"[INFO] 風險分析報告已儲存到: {output_path}")

    except Exception as e:
        print(f"[FATAL] 程式執行失敗: {e}")
        exit(1)