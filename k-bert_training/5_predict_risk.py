import torch
import os
import json
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from typing import List, Dict, Any

# --- 設定路徑 (必須與 3_kbert_event_detector.py 中的新路徑一致) ---
MODEL_OUTPUT_DIR = r"D:\senior_project\code\k-bert_training\kbert_model_output_multilabel"
MAX_SEQ_LENGTH = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定義四種風險的名稱 (與訓練時的標籤順序保持一致)
RISK_NAMES = [
    "Fall", 
    "Walk_with_memory_loss", 
    "Fall_with_climb", 
    "Run_with_disorientation"
]
# 預測的閾值 (高於此機率即判斷為危險)
THRESHOLD = 0.5 

# --- 輔助函式：將動作列表轉為文本 (保持不變) ---
def actions_to_text(actions: List[str]) -> str:
    """將動作列表轉換為 K-BERT 可接受的文本序列。"""
    sentences = []
    for action in actions:
        parts = action.split('_')
        action_name_parts = [part for part in parts if not part.isdigit() and part != '']
        action_name = '_'.join(action_name_parts)
        
        if action_name:
             sentences.append(f"The person {action_name.lower().replace('_', ' ')}")
        else:
             sentences.append(f"The person performs action {action.lower().replace('_', ' ')}")

    return ". ".join(sentences) + "."


# --- 核心預測函式 (適用於多標籤分類) ---
def predict_sequence(action_sequence: List[str], tokenizer: BertTokenizer, model: BertForSequenceClassification):
    """
    載入模型並預測給定動作序列的四種特定危險等級。
    返回: 預測向量 (0/1), 機率向量 (0~1)
    """
    
    # 1. 數據準備 (文本轉換)
    input_text = actions_to_text(action_sequence)
    
    # 2. Tokenization
    encoding = tokenizer.encode_plus(
        input_text,
        max_length=MAX_SEQ_LENGTH,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    # 3. 進行預測
    with torch.no_grad():
        inputs = {
            'input_ids': encoding['input_ids'].to(device),
            'attention_mask': encoding['attention_mask'].to(device),
            'token_type_ids': encoding['token_type_ids'].to(device),
        }
        
        outputs = model(**inputs)
        logits = outputs.logits # logits 形狀: (1, 4)
        
        # 關鍵：多標籤分類必須使用 Sigmoid 函式，將 logits 轉換為 0 到 1 的機率
        probabilities = torch.sigmoid(logits).squeeze().cpu().numpy() # shape (4,)
        
        # 關鍵：使用閾值 0.5 判斷每個標籤是 0 還是 1
        prediction_vector = (probabilities >= THRESHOLD).astype(int) # shape (4,)

    return prediction_vector.tolist(), probabilities.tolist(), input_text

# --- 主執行函式：批量處理 JSON 檔案 ---
def batch_predict_from_dir(target_dir: str):
    
    # 1. 載入模型 (在迴圈外先載入一次)
    if not os.path.exists(MODEL_OUTPUT_DIR):
        print(f"錯誤：找不到模型檔案於 {MODEL_OUTPUT_DIR}。請先運行 3_kbert_event_detector.py 並確保使用正確的輸出目錄。")
        return

    try:
        tokenizer = BertTokenizer.from_pretrained(MODEL_OUTPUT_DIR)
        model = BertForSequenceClassification.from_pretrained(MODEL_OUTPUT_DIR)
        model.to(device)
        model.eval()
        print("多標籤模型載入完成。開始批量預測...")
    except Exception as e:
        print(f"模型載入失敗，請檢查模型檔案是否為 {MODEL_OUTPUT_DIR}：{e}")
        return

    # 2. 遍歷資料夾中的所有 JSON 檔案
    if not os.path.exists(target_dir):
        print(f"錯誤：找不到指定的資料夾路徑: {target_dir}")
        return
        
    json_files = [f for f in os.listdir(target_dir) if f.endswith('.json')]
    
    if not json_files:
        print(f"目標資料夾 '{target_dir}' 中沒有找到 .json 檔案。")
        
    all_results = []
    
    for filename in json_files:
        file_path = os.path.join(target_dir, filename)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_json_data = json.load(f)
                
            # 提取動作序列
            activities = raw_json_data['data']['activities']
            
            # 進行預測
            prediction_vector, probabilities_vector, input_text = predict_sequence(activities, tokenizer, model)
            
            # 組織結果
            specific_risks = {RISK_NAMES[i]: prediction_vector[i] for i in range(len(RISK_NAMES))}
            
            final_output = {
                "file_id": raw_json_data['data']['id'],
                "action_count": len(activities),
                "prediction_vector": prediction_vector,
                "probabilities_vector": [f"{p:.4f}" for p in probabilities_vector], # 格式化機率
                "specific_risks": specific_risks
            }
            
            all_results.append(final_output)
            
            # 輸出每個檔案的結果
            risky_count = sum(prediction_vector)
            risk_summary = ", ".join([name for name, val in specific_risks.items() if val == 1])
            if risky_count == 0:
                 summary = "安全"
            else:
                 summary = f"{risky_count}種風險 ({risk_summary})"

            print(f"檔案: {filename} -> 預測: {summary}")
            
        except Exception as e:
            print(f"處理檔案 {filename} 時發生錯誤: {e}")
            
    # 3. 輸出總結
    print("\n--- 批量預測總結 (詳細) ---")
    for res in all_results:
        print(f"ID: {res['file_id']} | 動作數: {res['action_count']}")
        print(f"  機率: {res['probabilities_vector']}")
        print(f"  預測: {res['prediction_vector']} -> {res['specific_risks']}")


# --- 執行入口 (已簡潔化) ---
if __name__ == "__main__":
    # 設置目標資料夾 (請確認您的路徑是否正確)
    TARGET_DIR = r"E:\KGRC-RDF-kgrc4si\CompleteData\Episodes"
    
    # 這裡使用您上傳的檔案作為測試範例 (如果目標資料夾不存在)
    # 如果您已成功訓練多標籤模型，請確保 TARGET_DIR 存在
    
    # 假設您提供的檔案是該資料夾下的其中一個
    if not os.path.exists(TARGET_DIR):
        print("警告：指定的批量預測資料夾不存在，使用單一上傳檔案進行測試。")
        example_json_data = {
            "statusCode":200,"method":"GET","message":"Success",
            "data":{
                "id":"scene1_Day3_TEST",
                "title":"Day3",
                "scene":1,
                "activities":[
                    "Get_out_of_bed1",
                    "Walk_with_memory_loss6", # WML: 危險
                    "Put_slippers_in_closet1", 
                    "Walk_with_memory_loss5", # WML: 危險
                    "Cook_potato_using_microwave1", # 潛在 Fall 或其他風險
                    "Throw_trash3",
                    "Fall_backward1" # Fall: 危險
                ]
            }
        }
        
        # 使用單一測試數據進行預測
        try:
            tokenizer = BertTokenizer.from_pretrained(MODEL_OUTPUT_DIR)
            model = BertForSequenceClassification.from_pretrained(MODEL_OUTPUT_DIR)
            model.to(device)
            model.eval()
            
            prediction_vector, probabilities_vector, _ = predict_sequence(example_json_data['data']['activities'], tokenizer, model)
            
            print("\n--- 單一檔案測試結果 ---")
            print(f"動作: {', '.join(example_json_data['data']['activities'])}")
            print(f"機率: {probabilities_vector}")
            print(f"預測: {prediction_vector}")
            print(f"風險類型: { {RISK_NAMES[i]: prediction_vector[i] for i in range(len(RISK_NAMES))} }")

        except Exception as e:
            print(f"單一測試失敗，請確認多標籤模型已訓練完成：{e}")
    else:
        # 執行批量預測
        batch_predict_from_dir(TARGET_DIR)