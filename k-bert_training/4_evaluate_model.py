import torch
import json
import os
import random
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# 確保您的自定義數據載入器在同一資料夾
from kbert_custom_dataloader import KbertDataset, create_kbert_dataloader 

# --- 設定路徑 (必須與訓練腳本保持一致) ---
TRAIN_DATA_PATH = r"D:\senior_project\code\k-bert_training\kbert_train_data.jsonl" 
# 必須指向您多標籤模型訓練時儲存的位置
MODEL_OUTPUT_DIR = r"D:\senior_project\code\k-bert_training\kbert_model_output_multilabel" 
PRE_TRAINED_MODEL = 'bert-base-uncased' 

# --- 參數設定 ---
MAX_SEQ_LENGTH = 128
BATCH_SIZE = 16
TRAIN_RATIO = 0.8 
THRESHOLD = 0.5 # 多標籤判斷閾值
NUM_LABELS = 4  # 標籤數量

# 定義四種風險的名稱
RISK_NAMES = [
    "Fall", 
    "Walk_with_memory_loss", 
    "Fall_with_climb", 
    "Run_with_disorientation"
]

def load_and_split_data(filepath: str, train_ratio: float):
    """載入 JSONL 數據並劃分為訓練集和驗證集。"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
            
    random.seed(42) # 確保與訓練時的分割一致
    random.shuffle(data)

    split_point = int(len(data) * train_ratio)
    # val_data 是訓練時未見過的 20% 數據
    val_data = data[split_point:] 
    return val_data

def evaluate_model():
    
    # 1. 數據準備：載入驗證集
    val_data = load_and_split_data(TRAIN_DATA_PATH, TRAIN_RATIO)
    
    # 2. 載入多標籤模型和 Tokenizer
    try:
        tokenizer = BertTokenizer.from_pretrained(MODEL_OUTPUT_DIR)
        # 載入模型時必須指定 num_labels=4
        model = BertForSequenceClassification.from_pretrained(
            MODEL_OUTPUT_DIR, 
            num_labels=NUM_LABELS
        )
    except Exception as e:
        print(f"錯誤：模型載入失敗，請檢查 {MODEL_OUTPUT_DIR} 檔案是否存在且訓練腳本已執行：{e}")
        return
    
    # 3. 創建 DataLoader
    val_dataloader = create_kbert_dataloader(val_data, tokenizer, MAX_SEQ_LENGTH, BATCH_SIZE)
    
    # 4. 設置評估模式
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval() # 設置為評估模式

    all_predictions = []
    all_true_labels = []

    print(f"開始評估多標籤模型 (共 {len(val_data)} 條驗證數據)...")
    
    # 5. 評估迴圈 (Evaluation Loop)
    with torch.no_grad(): # 評估時不需要計算梯度
        for batch in tqdm(val_dataloader):
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device),
                'token_type_ids': batch['token_type_ids'].to(device),
            }
            # labels 必須是 float 類型
            labels = batch['labels'].cpu().numpy() # shape (Batch_Size, 4)
            
            outputs = model(**inputs)
            logits = outputs.logits # shape (Batch_Size, 4)
            
            # 關鍵修改：多標籤預測邏輯
            # 1. Sigmoid 轉換為機率
            probabilities = torch.sigmoid(logits) 
            # 2. 應用閾值 (Threshold) 轉換為 0/1 預測
            predictions = (probabilities >= THRESHOLD).cpu().numpy().astype(int) # shape (Batch_Size, 4)
            
            all_predictions.extend(predictions)
            all_true_labels.extend(labels)

    # 確保 NumPy 陣列格式正確
    all_predictions = np.array(all_predictions)
    all_true_labels = np.array(all_true_labels)

    # 6. 計算性能指標 (多標籤適用)
    
    # micro: 計算總體 TP, FP, FN 來得到指標，忽略標籤不平衡
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
        all_true_labels, 
        all_predictions, 
        average='micro', 
    )
    # macro: 計算每個標籤的指標後取平均，重視稀有標籤的性能
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        all_true_labels, 
        all_predictions, 
        average='macro', 
    )
    
    # 7. 輸出詳細結果
    print("\n\n--- 模型評估結果 (多標籤分類) ---")
    print(f"總驗證樣本數: {len(val_data)}")
    
    # 每個標籤的獨立 F1-Score
    label_f1 = precision_recall_fscore_support(
        all_true_labels, 
        all_predictions, 
        average=None, 
    )[2]
    
    print("\n[ 各別標籤 F1-Score ]")
    for i, name in enumerate(RISK_NAMES):
        # 這裡需要處理可能出現的 nan 情況 (如果某個標籤在驗證集中完全沒有出現)
        f1_score_val = label_f1[i] if not np.isnan(label_f1[i]) else 0.0
        print(f"  {name:<25}: {f1_score_val:.4f}")

    print("\n[ 總體指標 (Average Metrics) ]")
    print(f"Micro F1-Score: {micro_f1:.4f} (總體性能)")
    print(f"Macro F1-Score: {macro_f1:.4f} (考慮標籤平衡後的性能)")
    print(f"Micro Precision: {micro_precision:.4f}")
    print(f"Micro Recall: {micro_recall:.4f}")
    
    print("\n多標籤模型評估完成。")

# --- 執行評估 ---
if __name__ == "__main__":
    try:
        evaluate_model()
    except Exception as e:
        print(f"模型評估失敗，請檢查路徑和多標籤數據格式：{e}")